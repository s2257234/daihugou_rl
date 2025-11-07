"""AlphaZero 大富豪 学習トレーナ (初期版)

目的:
  - 自己対局によるデータ収集
  - リプレイバッファへの保存
  - モデルの定期学習 (train_step 呼び出し)
  - チェックポイント保存

前提:
  - 環境クラス: DaifugoSimpleEnv ( game 属性に進行状態を保持 )
  - エージェント: AlphaZeroAgent (agents.drl_agent.AlphaZeroAgent)
  - モデル: PolicyValueNet (agents.models.PolicyValueNet)

制約:
  - 現在 後続で損失計算 / 逆伝播を実装する。
  - 複数エージェント(4人)分のデータ管理は簡易。各 AlphaZeroAgent が自分のバッファを内部保持する。

使い方(例):
  from agents.drl_agent import AlphaZeroAgent
  from agents.models import PolicyValueNet
  from agents.config import ALPHA_ZERO_CONFIG
  from game.environment import DaifugoSimpleEnv

  trainer = Trainer(config=ALPHA_ZERO_CONFIG)
  trainer.setup()
  trainer.self_play(num_episodes=10)
  trainer.train_updates(num_updates=5)

TODO:
  - value ターゲット: 『次に上がるプレイヤー』フェーズ分類 (最終順位報酬なし)
  - 複数プレイヤー視点の value -> root 視点整合
  - 温度スケジューリング (中盤以降 temperature -> 0)
  - ログ収集 / TensorBoard 対応
"""
from __future__ import annotations

import os
import random
import time
from typing import Dict, Any, List
import concurrent.futures
from dataclasses import dataclass
import multiprocessing as mp
import argparse
import threading
import queue as _queue

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.drl_agent import AlphaZeroAgent
from agents.replay_buffer import ReplayBuffer
from agents.models import PolicyValueNet
from agents.config import ALPHA_ZERO_CONFIG
from utils.logger import TrainingLogger

try:
    from game.environment import DaifugoSimpleEnv
except ImportError as e:  # pragma: no cover
    raise ImportError("DaifugoSimpleEnv が見つかりません。game.environment を確認してください") from e


# ======================================================
# 並列自己対局: ワーカープロセス用エントリ関数 (Windows spawn 対応)
# ======================================================
def _selfplay_worker_entry(worker_id: int, episodes: int, config: Dict[str, Any], model_path: str):
    """ワーカープロセスで自己対局を複数回実行し、確定サンプルを返す。

    戻り値: {"samples": List[Dict], "episodes": int, "worker": int}
    """
    # まず CPU スレッド関連の環境変数を設定（numpy/MKL などの初期化前に行う）
    try:
        tn = int(config.get("torch_num_threads_workers", 0) or 0)
        if tn > 0:
            os.environ["OMP_NUM_THREADS"] = str(tn)
            os.environ["MKL_NUM_THREADS"] = str(tn)
            os.environ["OPENBLAS_NUM_THREADS"] = str(tn)
            os.environ["NUMEXPR_NUM_THREADS"] = str(tn)
        itn = int(config.get("torch_num_interop_threads_workers", 0) or 0)
        # interop は環境変数なし（PyTorch API で設定）
    except Exception:
        pass

    # 乱数初期化（プロセス毎にズラす）
    try:
        seed = int(config.get("seed", 42)) + 10000 * int(worker_id)
    except Exception:
        seed = 42 + 10000 * int(worker_id)
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

    # CPU スレッド数の制御（過剰スレッド競合の回避）
    try:
        tn = int(config.get("torch_num_threads_workers", 0) or 0)
        itn = int(config.get("torch_num_interop_threads_workers", 0) or 0)
        if tn > 0 or itn > 0:
            import torch as _t
            if tn > 0:
                _t.set_num_threads(tn)
            if itn > 0 and hasattr(_t, "set_num_interop_threads"):
                try:
                    _t.set_num_interop_threads(itn)
                except Exception:
                    pass
            try:
                _intra = _t.get_num_threads()
                _interop = _t.get_num_interop_threads() if hasattr(_t, "get_num_interop_threads") else -1
                print(f"[threads][worker {worker_id}] intra={_intra} interop={_interop} OMP={os.environ.get('OMP_NUM_THREADS')} MKL={os.environ.get('MKL_NUM_THREADS')}")
            except Exception:
                pass
    except Exception:
        pass

    # モデル読込（CPU 推奨） + フル特徴量対応
    from agents.models import PolicyValueNet as _PVN
    device = config.get("selfplay_worker_device", "cpu")
    # 共有CUDAテンソルのIPC警告を避けるため、既定でワーカーはCPUに固定
    # 明示的に無効化したい場合は config["force_worker_cpu"]=False
    if bool(config.get("force_worker_cpu", True)):
        device = "cpu"
    use_full = bool(config.get("use_full_features", False))
    try:
        model = _PVN.load(model_path, map_location=device)
        # 旧 ckpt で use_full_features が欠落しているが config が True の場合再構築
        if use_full and not getattr(model, 'use_full_features', False):
            # 後で環境初期化後に再構築
            pass
    except Exception:
        # フォールバック: 仮モデル (フル時は base_dim で後から再構築)
        try:
            model = _PVN(
                max_policy_size=config.get("max_policy_size", 128),
                hidden_size=config.get("hidden_size", 128),
                num_players=config.get("num_players", 4),
                device=device,
                use_full_features=use_full,
                full_feature_dim=(56 * config.get("num_players", 4) + 22) if use_full else None,
            )
        except Exception:
            # 最低限簡易
            model = _PVN(
                max_policy_size=config.get("max_policy_size", 128),
                hidden_size=config.get("hidden_size", 128),
                num_players=config.get("num_players", 4),
                device=device,
            )

    # エージェントと環境を構築（共有リプレイは使わずローカルに蓄積）
    num_players = int(config.get("num_players", 4))
    agents: List[AlphaZeroAgent] = []
    for pid in range(num_players):
        ag = AlphaZeroAgent(player_id=pid, model=model, config=dict(config))
        # ワーカー内では共有リプレイを無効化し、ローカル list に保存させる
        try:
            ag._use_shared = False
            ag.replay_buffer = []
        except Exception:
            pass
        agents.append(ag)
    env = DaifugoSimpleEnv(num_players=num_players, agent_classes=None)
    env.agents = agents
    for ag in agents:
        if hasattr(ag, 'set_env_ref'):
            ag.set_env_ref(env)
        if hasattr(ag, 'logger'):
            ag.logger = None

    # フル特徴量なら最初の reset 後に次元確定して再構築
    if use_full:
        try:
            env.reset()
            probe_state = agents[0]._extract_state(env)
            full_dim = probe_state.get('full_input_dim')
            if full_dim and (not getattr(model, 'use_full_features', False) or getattr(model, 'backbone', None) and getattr(model.backbone[0], 'in_features', None) != full_dim):
                model = _PVN(
                    max_policy_size=config.get("max_policy_size", 128),
                    hidden_size=config.get("hidden_size", 128),
                    num_players=config.get("num_players", 4),
                    device=device,
                    use_full_features=True,
                    full_feature_dim=full_dim,
                )
                for ag in agents:
                    ag.set_model(model)
        except Exception:
            print("[WARN][worker] full feature rebuild failed, fallback simple")

    # エピソード実行ループ（Trainer._play_one_episode とほぼ同等の縮約版）
    total_episodes = int(episodes)
    for ep in range(total_episodes):
        if hasattr(env, 'reset'):
            env.reset()
        for ag in agents:
            if hasattr(ag, 'reset_episode'):
                ag.reset_episode()
        step_count = 0
        prev_rankings: List[int] = list(getattr(env.game, "rankings", []))
        max_steps = int(config.get("max_episode_steps", 1000))
        while not getattr(env.game, "done", False):
            if step_count >= max_steps:
                break
            cur_pid = env.game.turn
            ag = agents[cur_pid]
            action = ag.select_action(env, training=True)
            try:
                env.step(external_action=action)
            except TypeError:
                env.step(action)
            step_count += 1
            # フェーズ確定時処理
            current_rankings: List[int] = list(getattr(env.game, "rankings", []))
            if len(current_rankings) > len(prev_rankings):
                new_winners = current_rankings[len(prev_rankings):]
                for winner_id in new_winners:
                    for az in agents:
                        was_active = az.player_id not in prev_rankings
                        az.finalize_phase(winner_player_id=winner_id, was_active=was_active)
                prev_rankings = current_rankings
        # 未確定フェーズを 0 で確定し、ゲーム終端フック
        for az in agents:
            az.flush_unfinished_phase()
            az.finalize_game()

    # 確定サンプルを収集して返す
    out_samples: List[Dict[str, Any]] = []
    for az in agents:
        try:
            for s in az.replay_buffer:
                if isinstance(s, dict) and s.get("value") is not None:
                    out_samples.append(dict(s))
        except Exception:
            pass
    return {"samples": out_samples, "episodes": total_episodes, "worker": worker_id}


# ======================================================
# 非同期評価ゲート: 子プロセス用ワーカー関数
# ======================================================
def _gate_worker_eval(candidate_ckpt_path: str, baseline_ckpt_path: str, cfg: Dict[str, Any], games: int, seed: int | None, device: str = "cpu") -> Dict[str, Any]:
    """サブプロセスで評価を行い結果を返す（設定は呼び出し元の config を尊重）。"""
    # 遅延 import（サブプロセス側）
    from agents.models import PolicyValueNet as _PVN
    from evaluation.gating import evaluate_candidate

    # モデル読み込み（CPU推奨）
    cand = _PVN.load(candidate_ckpt_path, map_location=device)
    base = _PVN.load(baseline_ckpt_path, map_location=device)
    try:
        if hasattr(cand, 'to'):
            cand.to(device)
    except Exception:
        pass
    try:
        if hasattr(base, 'to'):
            base.to(device)
    except Exception:
        pass
    # 評価実行（gating 側の _override_eval_config のみ適用。ここでは cfg を尊重）
    res = evaluate_candidate(cand, base, cfg, games=int(games), seed=seed)
    return res

def _gate_worker_eval_proc(candidate_ckpt_path: str,
                           baseline_ckpt_path: str,
                           cfg: Dict[str, Any],
                           games: int,
                           seed: int | None,
                           device: str,
                           out_q):
    """mp.Process ターゲット関数: 結果を Queue へ put して終了。"""
    try:
        res = _gate_worker_eval(candidate_ckpt_path, baseline_ckpt_path, cfg, games, seed, device)
    except Exception as e:
        try:
            out_q.put({"error": f"{type(e).__name__}: {e}"}, block=False)
        except Exception:
            pass
        return
    try:
        out_q.put(res, block=False)
    except Exception:
        pass


# ======================================================
# 常駐自己対局: 並行学習用ワーカープロセス (Producer)
# ======================================================
def _selfplay_daemon_worker(worker_id: int,
                            config: Dict[str, Any],
                            model_path: str,
                            sample_queue,  # mp.Queue
                            event_queue,   # mp.Queue ("ep_done" 通知など + 制御メッセージ)
                            stop_event,    # mp.Event 全体停止
                            control_queue=None  # 親→子 制御 (flush/exit 指示)
                            ):   # mp.Event
    """常駐で自己対局を繰り返し、確定サンプルを逐次 sample_queue へ push する。

    event_queue へはエピソード完了ごとに ("ep_done", 1) を送る。
    モデル更新は model_path の mtime を監視して自動リロードする（数エピソード毎にチェック）。
    """
    # まず CPU スレッド関連の環境変数を設定（numpy/MKL などの初期化前に行う）
    try:
        tn = int(config.get("torch_num_threads_workers", 0) or 0)
        if tn > 0:
            os.environ["OMP_NUM_THREADS"] = str(tn)
            os.environ["MKL_NUM_THREADS"] = str(tn)
            os.environ["OPENBLAS_NUM_THREADS"] = str(tn)
            os.environ["NUMEXPR_NUM_THREADS"] = str(tn)
        itn = int(config.get("torch_num_interop_threads_workers", 0) or 0)
    except Exception:
        pass

    # 乱数初期化（プロセス毎にズラす）
    try:
        seed = int(config.get("seed", 42)) + 10000 * int(worker_id)
    except Exception:
        seed = 42 + 10000 * int(worker_id)
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

    # CPU スレッド数の制御（過剰スレッド競合の回避）
    try:
        tn = int(config.get("torch_num_threads_workers", 0) or 0)
        itn = int(config.get("torch_num_interop_threads_workers", 0) or 0)
        if tn > 0 or itn > 0:
            import torch as _t
            if tn > 0:
                _t.set_num_threads(tn)
            if itn > 0 and hasattr(_t, "set_num_interop_threads"):
                try:
                    _t.set_num_interop_threads(itn)
                except Exception:
                    pass
            try:
                _intra = _t.get_num_threads()
                _interop = _t.get_num_interop_threads() if hasattr(_t, "get_num_interop_threads") else -1
                print(f"[threads][daemon {worker_id}] intra={_intra} interop={_interop} OMP={os.environ.get('OMP_NUM_THREADS')} MKL={os.environ.get('MKL_NUM_THREADS')}")
            except Exception:
                pass
    except Exception:
        pass

    # モデル読込（CPU 推奨） + フル特徴量対応
    from agents.models import PolicyValueNet as _PVN
    device = config.get("selfplay_worker_device", "cpu")
    if bool(config.get("force_worker_cpu", True)):
        device = "cpu"
    use_full = bool(config.get("use_full_features", False))
    try:
        model = _PVN.load(model_path, map_location=device)
        if use_full and not getattr(model, 'use_full_features', False):
            pass
    except Exception:
        try:
            model = _PVN(
                max_policy_size=config.get("max_policy_size", 128),
                hidden_size=config.get("hidden_size", 128),
                num_players=config.get("num_players", 4),
                device=device,
                use_full_features=use_full,
                full_feature_dim=(56 * config.get("num_players", 4) + 22) if use_full else None,
            )
        except Exception:
            model = _PVN(
                max_policy_size=config.get("max_policy_size", 128),
                hidden_size=config.get("hidden_size", 128),
                num_players=config.get("num_players", 4),
                device=device,
            )

    # モデル更新監視
    def _get_mtime(p):
        try:
            return os.path.getmtime(p)
        except Exception:
            return 0.0
    last_mtime = _get_mtime(model_path)
    # 過去モデルプール（親が保存したスナップショット）
    pool_dir = os.path.join(config.get("checkpoint_dir", "checkpoints"), "_pool")
    learning_pid = int(config.get("learning_player_id", 0) or 0)
    enable_mix = bool(config.get("keep_previous_model_opponent", False)) and int(config.get("previous_model_mix_players", 0) or 0) > 0
    mix_players = int(config.get("previous_model_mix_players", 0) or 0)
    mix_interval = int(config.get("opponent_mix_interval_episodes", 0) or 0)
    ep_since_mix = 0

    def _list_pool_files():
        try:
            if not os.path.isdir(pool_dir):
                return []
            files = [os.path.join(pool_dir, f) for f in os.listdir(pool_dir) if f.endswith('.pt')]
            # 新しい順（将来のポリシーで優先したい場合）だがランダムに選ぶので順序は重要ではない
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return files
        except Exception:
            return []

    def _assign_opponents_from_pool():
        """学習プレイヤー以外の一部に過去モデルを割当てる。プールが空なら全員最新で継続。"""
        try:
            # 学習プレイヤーは常に最新モデル
            try:
                ag_learner = agents[learning_pid]
                if hasattr(ag_learner, 'set_model'):
                    ag_learner.set_model(model)
            except Exception:
                pass

            if not enable_mix or mix_players <= 0:
                return
            pool_files = _list_pool_files()
            if not pool_files:
                return
            # 候補プレイヤー
            cand = [i for i in range(num_players) if i != learning_pid]
            if not cand:
                return
            random.shuffle(cand)
            selected = cand[:mix_players]
            # まず全員を最新に戻す（選ばれた者だけ後で過去モデルを当てる）
            for i in cand:
                try:
                    if hasattr(agents[i], 'set_model'):
                        agents[i].set_model(model)
                except Exception:
                    pass
            # 選ばれた相手にプールからランダム割当
            for idx in selected:
                try:
                    path = random.choice(pool_files)
                    opp_model = _PVN.load(path, map_location=device)
                    if hasattr(agents[idx], 'set_model'):
                        agents[idx].set_model(opp_model)
                except Exception:
                    # 読込失敗時は最新モデルのまま
                    pass
        except Exception:
            pass

    # エージェントと環境を構築（共有バッファは使用せず、サンプルは Queue へ）
    num_players = int(config.get("num_players", 4))
    agents: List[AlphaZeroAgent] = []
    worker_zero_buffer = bool(config.get("worker_zero_buffer", False))
    for pid in range(num_players):
        ag = AlphaZeroAgent(player_id=pid, model=model, config=dict(config))
        # ワーカー内では共有リプレイを使わない
        try:
            ag._use_shared = False
            # worker_zero_buffer が True の場合は None、False の場合は空リスト
            ag.replay_buffer = None if worker_zero_buffer else []
            ag.logger = None
        except Exception:
            pass
        agents.append(ag)
    env = DaifugoSimpleEnv(num_players=num_players, agent_classes=None)
    env.agents = agents
    for ag in agents:
        if hasattr(ag, 'set_env_ref'):
            ag.set_env_ref(env)
        if hasattr(ag, 'logger'):
            ag.logger = None

    max_steps = int(config.get("max_episode_steps", 1000))
    check_interval_episodes = int(config.get("concurrent_model_check_every", 5) or 5)
    ep_since_check = 0

    # 初期の対戦相手割当（プールがあれば活用）
    if enable_mix and mix_players > 0:
        _assign_opponents_from_pool()

    # フル特徴量: 初期 reset で次元確定し必要なら再構築
    if use_full:
        try:
            env.reset()
            probe_state = agents[0]._extract_state(env)
            full_dim = probe_state.get('full_input_dim')
            if full_dim and (not getattr(model, 'use_full_features', False) or getattr(model, 'backbone', None) and getattr(model.backbone[0], 'in_features', None) != full_dim):
                model = _PVN(
                    max_policy_size=config.get("max_policy_size", 128),
                    hidden_size=config.get("hidden_size", 128),
                    num_players=config.get("num_players", 4),
                    device=device,
                    use_full_features=True,
                    full_feature_dim=full_dim,
                )
                for ag in agents:
                    ag.set_model(model)
        except Exception:
            print("[WARN][daemon_worker] full feature rebuild failed, fallback simple")

    # 制御: シャードフラッシュ用ローカル関数
    def _flush_local_buffers_to_queue():
        """未送信確定サンプルを sample_queue へ流す (重複送信防止のため flush 後にクリア)。"""
        try:
            for az in agents:
                buf = getattr(az, 'replay_buffer', [])
                if not buf:
                    continue
                for s in list(buf):  # コピー上を走査
                    if isinstance(s, dict) and s.get('value') is not None:
                        try:
                            sample_queue.put(s, block=True)
                        except Exception:
                            break
                try:
                    buf.clear()
                except Exception:
                    pass
        except Exception:
            pass

    def _save_shard_if_any(reason: str):
        """ローカル(ワーカー内)にまだ残る確定サンプル(念のため)をシャードファイルに保存。
        Queue 送信前後で競合しうるが、目的はロス最小なので多少の重複は許容。"""
        try:
            pending: List[dict] = []
            for az in agents:
                for s in getattr(az, 'replay_buffer', []) or []:
                    if isinstance(s, dict) and s.get('value') is not None:
                        pending.append(s)
            if not pending:
                return 0
            shard_dir = os.path.join(config.get('checkpoint_dir', 'checkpoints'), 'replay_shards')
            os.makedirs(shard_dir, exist_ok=True)
            ts = int(time.time()*1000)
            fname = f"worker{worker_id}_pid{os.getpid()}_{ts}_{len(pending)}_{reason}.joblib"
            tmp = os.path.join(shard_dir, fname + '.tmp')
            final = os.path.join(shard_dir, fname)
            try:
                import joblib as _jb
                _jb.dump(pending, tmp, compress=3)
                os.replace(tmp, final)
                return len(pending)
            except Exception:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
        except Exception:
            pass
        return 0

    while not (stop_event.is_set()):
        # 制御キューのポーリング (ノンブロッキング)
        if control_queue is not None:
            try:
                ctrl = control_queue.get_nowait()
            except Exception:
                ctrl = None
            if ctrl == 'FLUSH_AND_EXIT':
                # 1) まず queue へ flush
                _flush_local_buffers_to_queue()
                # 2) 残っていればシャード保存
                _save_shard_if_any('grace')
                try:
                    event_queue.put(('worker_exit', worker_id), block=False)
                except Exception:
                    pass
                return  # グレースフル終了
        # 必要ならモデル更新
        if ep_since_check >= check_interval_episodes:
            ep_since_check = 0
            cur_mtime = _get_mtime(model_path)
            if cur_mtime > last_mtime:
                try:
                    model = _PVN.load(model_path, map_location=device)
                    # 学習プレイヤーのみ最新へ（対戦相手の過去モデル割当は維持）
                    try:
                        ag_learner = agents[learning_pid]
                        if hasattr(ag_learner, 'set_model'):
                            ag_learner.set_model(model)
                    except Exception:
                        pass
                except Exception:
                    pass
                last_mtime = cur_mtime
                # モデル更新後に、必要なら対戦相手の再割当を行う（次のミックスタイミングで行う）
                # 即時適用したい場合は以下を有効化
                # if enable_mix and mix_players > 0:
                #     _assign_opponents_from_pool()

        # 1 エピソード実行
        if hasattr(env, 'reset'):
            env.reset()
        for ag in agents:
            if hasattr(ag, 'reset_episode'):
                ag.reset_episode()
        step_count = 0
        prev_rankings: List[int] = list(getattr(env.game, "rankings", []))
        dbg_samples_before = [len(buf) for ag in agents if hasattr(ag, 'replay_buffer') and (buf := getattr(ag, 'replay_buffer')) is not None and hasattr(buf, '__len__')]
        while not getattr(env.game, "done", False):
            if step_count >= max_steps:
                break
            cur_pid = env.game.turn
            ag = agents[cur_pid]
            action = ag.select_action(env, training=True)
            try:
                env.step(external_action=action)
            except TypeError:
                env.step(action)
            step_count += 1
            # 軽量ハートビート: 長手局面でも親に進行を通知（詰まり診断用）
            if (step_count % 200) == 0:
                try:
                    event_queue.put(("hb", (worker_id, step_count)), block=False)
                except Exception:
                    pass
            # フェーズ確定処理
            current_rankings: List[int] = list(getattr(env.game, "rankings", []))
            if len(current_rankings) > len(prev_rankings):
                new_winners = current_rankings[len(prev_rankings):]
                for winner_id in new_winners:
                    for az in agents:
                        was_active = az.player_id not in prev_rankings
                        az.finalize_phase(winner_player_id=winner_id, was_active=was_active)
                try:
                    if config.get('debug_concurrent_worker', False):
                        # 勝者発生時点で現在のフェーズ確定サンプル数をログ
                        labeled_counts = []
                        total_counts = []
                        for az in agents:
                            buf = getattr(az, 'replay_buffer', [])
                            total_counts.append(len(buf) if buf is not None else 0)
                            lc = 0
                            for s in buf:
                                if isinstance(s, dict) and s.get('value') is not None:
                                    lc += 1
                            labeled_counts.append(lc)
                        print(f"[WORKER{worker_id}][phase] winners={new_winners} totals={total_counts} labeled={labeled_counts}")
                except Exception:
                    pass
                prev_rankings = current_rankings
        # エピソード終端処理
        for az in agents:
            az.flush_unfinished_phase()
            az.finalize_game()
        if config.get('debug_concurrent_worker', False):
            try:
                after_counts = []
                labeled_after = []
                for az in agents:
                    buf = getattr(az, 'replay_buffer', [])
                    after_counts.append(len(buf) if buf is not None else 0)
                    la = 0
                    for s in buf:
                        if isinstance(s, dict) and s.get('value') is not None:
                            la += 1
                    labeled_after.append(la)
                print(f"[WORKER{worker_id}][episode_end] steps={step_count} buf_total={after_counts} buf_labeled={labeled_after}")
            except Exception:
                pass

        # このエピソードで確定したサンプルを Queue へ送信
        for az in agents:
            try:
                buf = getattr(az, 'replay_buffer', None)
                if buf is None:
                    # 代替案: エージェントに _confirmed_samples というリストを追加し、
                    # finalize/flush 時にそこに移動してからクリアする。
                    # 
                    # 緊急対応: 各エージェントに _episode_confirmed_samples を追加
                    episode_samples = getattr(az, '_episode_confirmed_samples', [])
                    for s in episode_samples:
                        if isinstance(s, dict) and s.get("value") is not None:
                            sample_queue.put(s, block=True)
                    # クリア
                    if hasattr(az, '_episode_confirmed_samples'):
                        az._episode_confirmed_samples.clear()
                    continue
                for s in list(buf):
                    if isinstance(s, dict) and s.get("value") is not None:
                        sample_queue.put(s, block=True)
                buf.clear()  # 重複防止
            except Exception:
                pass

        # 進捗イベント
        try:
            event_queue.put(("ep_done", 1), block=True)
        except Exception:
            pass

        # --------------------------------------------------
        # 低頻度統計ログ: 平均手数/ep, p95手数, 実効シミュレーション数(平均/分布), early-stop率
        # 注意: 追加の設定キーは導入せず、固定間隔(50 ep)で送信
        # --------------------------------------------------
        try:
            bi = __import__('builtins')
            _epc = getattr(bi, '__worker_ep_done', 0) + 1  # type: ignore[attr-defined]
            setattr(bi, '__worker_ep_done', _epc)          # type: ignore[attr-defined]
            # エピソード手数を集計（各エージェントのmove_count合計 = 総手数）
            try:
                ep_moves = 0
                for az in agents:
                    ep_moves += int(getattr(az, 'move_count', 0) or 0)
            except Exception:
                ep_moves = 0
            try:
                mv_list = getattr(bi, '__worker_ep_moves', None)
                if mv_list is None:
                    mv_list = []
                    setattr(bi, '__worker_ep_moves', mv_list)
                mv_list.append(ep_moves)
                # サイズ制限（過剰メモリ回避）
                if len(mv_list) > 5000:
                    del mv_list[:len(mv_list)-5000]
            except Exception:
                pass

            if _epc % 200 == 0:
                # 分布ユーティリティ
                def _quantile(sorted_arr, q: float):
                    if not sorted_arr:
                        return None
                    q = max(0.0, min(1.0, float(q)))
                    idx = int(round((len(sorted_arr)-1) * q))
                    return float(sorted_arr[idx])
                # 手数統計
                mv_copy = list(getattr(bi, '__worker_ep_moves', []) or [])
                mv_avg = float(sum(mv_copy)/len(mv_copy)) if mv_copy else None
                mv_p95 = _quantile(sorted(mv_copy), 0.95) if mv_copy else None
                # 実効シミュレーション数 (直近窓から集計)
                sims_all = []
                total_moves = 0
                early_stops = 0
                try:
                    for az in agents:
                        vals = list(getattr(az, '_mcts_sims_recent', []) or [])
                        sims_all.extend(int(v) for v in vals)
                        total_moves += int(getattr(az, '_mcts_moves', 0) or 0)
                        early_stops += int(getattr(az, '_mcts_early_stop', 0) or 0)
                except Exception:
                    pass
                sims_avg = (float(sum(sims_all))/len(sims_all)) if sims_all else None
                sims_p50 = _quantile(sorted(sims_all), 0.50) if sims_all else None
                sims_p95 = _quantile(sorted(sims_all), 0.95) if sims_all else None
                es_rate = (float(early_stops)/float(total_moves)) if total_moves>0 else None
                payload = {
                    'wid': int(worker_id),
                    'ep': int(_epc),
                    'moves_avg': mv_avg,
                    'moves_p95': mv_p95,
                    'sims_avg': sims_avg,
                    'sims_p50': sims_p50,
                    'sims_p95': sims_p95,
                    'early_stop_rate': es_rate,
                    'sample_size': len(sims_all),
                }
                try:
                    event_queue.put(("play_stats", payload), block=False)
                except Exception:
                    pass
        except Exception:
            pass
        ep_since_check += 1
        # 過去モデルミックスの周期適用
        if enable_mix and mix_interval > 0:
            ep_since_mix += 1
            if ep_since_mix >= mix_interval:
                _assign_opponents_from_pool()
                ep_since_mix = 0

    # 終了時フラッシュ（念のため）
    try:
        event_queue.put(("worker_exit", worker_id), block=False)
    except Exception:
        pass


class Trainer:
    def __init__(self, config: Dict[str, Any] | None = None):
        # 設定読み込み
        self.config = dict(ALPHA_ZERO_CONFIG)
        if config:
            self.config.update(config)
        # 進捗表示オプション (大量エピソード時の簡潔出力)
        self.minimal_progress = self.config.get("minimal_progress", True)
        self.progress_update_interval = self.config.get("progress_update_interval", 100)
        # 進捗バー設定
        self.use_progress_bar = self.config.get("progress_bar", True)
        self.progress_bar_width = self.config.get("progress_bar_width", 40)
        # 動的バー用内部ステート
        self._last_progress_len = 0
        # ETA 改善用パラメータ (エピソード時間の指数移動平均)
        self.eta_alpha = self.config.get("eta_smoothing_alpha", 0.25)  # 0.0(なし)〜1.0(最新のみ)
        self.monotonic_eta = self.config.get("monotonic_eta", True)    # True なら残り時間を単調減少にクランプ
        self._eta_smooth = None  # 型: float|None (EMAされた 1 エピソード所要時間秒)
        self._eta_prev_remaining = None  # 前回表示した残り秒 (単調減少用)
        # メンバ初期化
        self.agents = []  # type: ignore[list]
        self.env = None
        self.model = None
        random.seed(self.config.get("seed", 42))
        # ウォームアップ設定
        self.warmup_episodes = self.config.get("warmup_episodes", 0)
        self.warmup_mix = self.config.get("warmup_mix", ["random", "rule", "random"])  # 学習エージェント以外の順番
        # 学習対象プレイヤーID
        self.learning_player_id = self.config.get("learning_player_id", 0)
        # ロガー (setup で初期化)
        self.logger = None  # type: ignore
        # 追加: 周期チェックポイント & 直前モデルミックス用
        self.ckpt_interval = self.config.get("checkpoint_interval_episodes", 0) or 0
        self.keep_prev_model = self.config.get("keep_previous_model_opponent", False)
        self.prev_model_mix_players = int(self.config.get("previous_model_mix_players", 0) or 0)
        self._previous_model = None  # type: ignore
        self._episodes_total_run = 0  # 累積自己対局回数 (複数 self_play 呼び出し対応)
        # モデル世代管理 (リプレイに過去モデル由来サンプルを混在保持)
        self.model_version = 0
        # --- 追加: 複数過去モデルプール ---
        # 過去モデルを複数保持し、対戦相手へランダム割当して探索多様性を向上させる。
        # keep_previous_model_opponent / previous_model_mix_players が有効な場合にのみ使用。
        self.past_models = []  # List[PolicyValueNet]
        self.past_model_pool_size = int(self.config.get("past_model_pool_size", 5))  # 保存上限 (古い順に削除)
        # 何エピソードごとに opponent へ再割当するか (0/未指定なら checkpoint タイミングでのみ)
        self.opponent_mix_interval = int(self.config.get("opponent_mix_interval_episodes", 0) or 0)
        
        # --- 非同期ゲート評価用 ---
        self._gate_pool = None  # deprecated: ProcessPoolExecutor (not used)
        self._gate_future = None  # deprecated: Future (not used)
        self._gate_proc = None
        self._gate_queue = None
        self._gate_candidate_path = None
        self._gate_baseline_path = None
        # ゲート最新結果（毎更新ログ用に保持）
        try:
            self._last_gate_threshold = float(self.config.get("eval_gate_threshold", 0.6) or 0.6)
        except Exception:
            self._last_gate_threshold = 0.6
        self._last_gate_result = None  # 型: Optional[Dict[str, Any]]
        # 評価ゲートの起動制御（学習更新回数ベース）
        self._last_gate_start_it = -1  # 直近の評価開始時点の train_it（未開始は -1）

    def _should_start_gate(self, train_it: int) -> bool:
        """評価ゲートを開始してよいかを学習更新回数で判定する。
        - eval_gate_start_after_updates: この回数に到達するまで起動しない
        - eval_gate_every_updates: 直近の起動からこの回数に満たない間は起動しない
        """
        try:
            start_after = int(self.config.get("eval_gate_start_after_updates", 0) or 0)
        except Exception:
            start_after = 0
        try:
            every = int(self.config.get("eval_gate_every_updates", 0) or 0)
        except Exception:
            every = 0
        if train_it < start_after:
            return False
        if every > 0 and self._last_gate_start_it >= 0 and (train_it - self._last_gate_start_it) < every:
            return False
        return True

    # -----------------------------------------------------
    # 準備
    # -----------------------------------------------------
    def setup(self):
        # まずメインプロセスのスレッド環境変数を先に設定（ライブラリ初期化前に反映させる）
        try:
            tn_main_env = int(self.config.get("torch_num_threads_main", 0) or 0)
            if tn_main_env > 0:
                os.environ["OMP_NUM_THREADS"] = str(tn_main_env)
                os.environ["MKL_NUM_THREADS"] = str(tn_main_env)
                os.environ["OPENBLAS_NUM_THREADS"] = str(tn_main_env)
                os.environ["NUMEXPR_NUM_THREADS"] = str(tn_main_env)
        except Exception:
            pass

        # デバイス決定とモデル生成
        device_cfg = self.config.get("device", "auto")
        if device_cfg == "auto":
            try:
                import torch  # lazy import to check availability
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                resolved_device = "cpu"
        else:
            resolved_device = device_cfg
        # メインプロセスの CPU スレッド数を制御（必要時）
        try:
            tn_main = int(self.config.get("torch_num_threads_main", 0) or 0)
            itn_main = int(self.config.get("torch_num_interop_threads_main", 0) or 0)
            if tn_main > 0 or itn_main > 0:
                import torch as _t
                if tn_main > 0:
                    _t.set_num_threads(tn_main)
                if itn_main > 0 and hasattr(_t, "set_num_interop_threads"):
                    try:
                        _t.set_num_interop_threads(itn_main)
                    except Exception:
                        pass
        except Exception:
            pass
        # 既存チェックポイントからの再開を試みる
        self._loaded_from_checkpoint = False
        latest_path = self.config.get("checkpoint_path", os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "policy_value_latest.pt"))
        try:
            if latest_path and os.path.isfile(latest_path):
                # 既存の最新 ckpt があれば、それを優先的にロード
                try:
                    self.model = PolicyValueNet.load(latest_path, map_location=resolved_device)
                    self._loaded_from_checkpoint = True
                    # 端末へも簡易表示（logger 初期化前のため）
                    print(f"[resume] loaded model from {latest_path}")
                except Exception as e:
                    print(f"[WARN] failed to load checkpoint {latest_path}: {e}")
        except Exception:
            pass

        # フル特徴量使用時は一旦ダミー生成し、後で初回状態から次元を取得して再構築する方式を避けるため
        use_full = self.config.get("use_full_features", False)
        if not self._loaded_from_checkpoint:
            if use_full:
                # 一時インスタンス (full_feature_dim 後で差し替え。仮に base 次元で作る) -> 後続で lazy resize
                # ただし簡易: 最初の環境 reset 後に full_input_dim を取得して再初期化
                self.model = PolicyValueNet(
                    max_policy_size=self.config["max_policy_size"],
                    hidden_size=self.config["hidden_size"],
                    num_players=self.config["num_players"],
                    device=resolved_device,
                    use_full_features=True,
                    full_feature_dim= 56 * self.config["num_players"] + 22  # 統一フォーマット (後で再構築可)
                )
            else:
                self.model = PolicyValueNet(
                    max_policy_size=self.config["max_policy_size"],
                    hidden_size=self.config["hidden_size"],
                    num_players=self.config["num_players"],
                    device=resolved_device,
                    use_full_features=False,
                )
        # ロガー生成
        self.logger = TrainingLogger(
            log_dir=self.config.get("log_dir", "logs"),
            use_tensorboard=self.config.get("enable_tensorboard", True),
            clear_existing=self.config.get("clear_logs_on_start", False),
            log_mcts_samples=not self.config.get("disable_mcts_log", False),
            config=self.config
        )
        # スレッド設定の実測値ログは削除（簡潔化）
        # 共有リプレイバッファ (use_shared_replay=true の場合)
        self.shared_replay = None
        if self.config.get("use_shared_replay", False):
            self.shared_replay = ReplayBuffer(
                maxlen=self.config.get("buffer_size", 50000),
                path=self.config.get("replay_path", "replay_buffer.joblib")
            )
        # 初期は全員学習エージェント (ウォームアップで差し替える)
        self.agents = []
        for i in range(self.config["num_players"]):
            ag = AlphaZeroAgent(player_id=i, model=self.model, config=self.config)
            # 初期世代を注入
            if hasattr(ag, 'model_version'):
                ag.model_version = self.model_version
            # 共有モードならリプレイ参照注入
            if self.shared_replay is not None:
                ag.replay_buffer = self.shared_replay
            self.agents.append(ag)
        # 環境生成 (内部の env.agents も後で差し替える可能性があるため agent_classes=None)
        self.env = DaifugoSimpleEnv(num_players=self.config["num_players"], agent_classes=None)
        # 直ちに環境の agents を学習エージェント群で上書き (ウォームアップ0の場合の不整合防止)
        self.env.agents = self.agents
        # フル特徴量モデル再構築 (初回状態から full_input_dim 取得)
        # ただし、既存チェックポイントから再開した場合は重みを保持するためスキップ
        if use_full and (not getattr(self, "_loaded_from_checkpoint", False)):
            try:
                _ = self.env.reset()
                sample_state = self.agents[0]._extract_state(self.env)
                full_dim = sample_state.get('full_input_dim')
                if full_dim is not None:
                    resolved_device2 = resolved_device
                    self.model = PolicyValueNet(
                        max_policy_size=self.config["max_policy_size"],
                        hidden_size=self.config["hidden_size"],
                        num_players=self.config["num_players"],
                        device=resolved_device2,
                        use_full_features=True,
                        full_feature_dim=full_dim,
                    )
                    # エージェントへ新モデルを再注入
                    for ag in self.agents:
                        ag.set_model(self.model)
                        ag.set_model_version(self.model_version)
                    if self.logger:
                        self.logger.log_text(f"[model] Rebuilt full-feature model input_dim={full_dim}")
            except Exception as e:
                print(f"[WARN] full feature model rebuild failed: {e}")
        elif use_full and getattr(self, "_loaded_from_checkpoint", False):
            # 再開時はロードしたモデルの入出力次元を尊重。必要なら _encode_state 側で pad/truncate。
            try:
                if self.logger:
                    self.logger.log_text("[resume] skip full-feature rebuild to keep loaded weights")
            except Exception:
                pass
        # フルモード時: 旧フォーマットサンプル浄化 (初期残存している可能性に備える)
        if self.config.get('use_full_features'):
            for ag in self.agents:
                try:
                    if isinstance(ag.replay_buffer, list) and ag.replay_buffer:
                        ag.replay_buffer = [s for s in ag.replay_buffer if s.get('feature_version',1)==1]
                except Exception:
                    pass
        # 既存メタデータとの整合性チェック (存在すれば) + モデル世代の復元
        try:
            meta_path = os.path.join(self.config['checkpoint_dir'], 'metadata.json')
            if os.path.isfile(meta_path):
                import json as _json
                with open(meta_path, 'r', encoding='utf-8') as f:
                    old_meta = _json.load(f)
                if old_meta.get('use_full_features') != self.config.get('use_full_features'):
                    print('[WARN] metadata.use_full_features differs from current config')
                # モデル世代の復元（存在すれば）
                try:
                    mv = int(old_meta.get('model_version'))
                    self.model_version = mv
                    # 既存エージェントへも反映
                    for ag in self.agents:
                        if hasattr(ag, 'model_version'):
                            ag.model_version = self.model_version
                        if hasattr(ag, 'set_model_version'):
                            try:
                                ag.set_model_version(self.model_version)
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass
        # Optimizer の再開: 学習エージェントの optimizer 状態を復元（存在すれば）
        try:
            opt_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'optimizer_latest.pt')
            if os.path.isfile(opt_path):
                ag0 = self.agents[self.learning_player_id]
                # Optimizer を用意してから読み込み
                try:
                    if hasattr(ag0, 'ensure_optimizer'):
                        ag0.ensure_optimizer()
                    if hasattr(ag0, 'load_optimizer'):
                        ok = ag0.load_optimizer(opt_path, map_location=resolved_device)
                        if self.logger:
                            self.logger.log_text(f"[resume] optimizer load {'ok' if ok else 'failed'} from {opt_path}")
                except Exception as e:
                    print(f"[WARN] optimizer load failed: {e}")
        except Exception:
            pass
        # Scheduler の再開: 学習エージェントの学習率スケジューラを復元（存在すれば）
        try:
            sch_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'scheduler_latest.pt')
            if os.path.isfile(sch_path):
                ag0 = self.agents[self.learning_player_id]
                try:
                    if hasattr(ag0, 'ensure_optimizer'):
                        ag0.ensure_optimizer()
                    if hasattr(ag0, 'ensure_scheduler'):
                        ag0.ensure_scheduler()
                    if hasattr(ag0, 'load_scheduler'):
                        ok = ag0.load_scheduler(sch_path)
                        if self.logger:
                            self.logger.log_text(f"[resume] scheduler load {'ok' if ok else 'failed'} from {sch_path}")
                except Exception as e:
                    print(f"[WARN] scheduler load failed: {e}")
        except Exception:
            pass
        # オプション: 今回の再開に限り Warmup をやり直す（ワンショット）
        try:
            if bool(self.config.get('resume_reset_warmup_once', False)):
                ag0 = self.agents[self.learning_player_id]
                if hasattr(ag0, 'reset_scheduler_warmup'):
                    done = ag0.reset_scheduler_warmup()
                    if self.logger:
                        self.logger.log_text(f"[resume] warmup reset {'ok' if done else 'failed'} (one-shot)")
                # プロセス内で一度だけ実行するためフラグを下ろす
                self.config['resume_reset_warmup_once'] = False
        except Exception:
            pass
        # 各エージェントへ環境参照を渡す (obs だけ渡される呼び出し互換のため)
        for ag in self.agents:
            if hasattr(ag, 'set_env_ref'):
                ag.set_env_ref(self.env)
            # 後方互換: エージェントが logger を受け取れるなら設定
            if hasattr(ag, 'logger'):
                ag.logger = self.logger
            # 念のため共有リプレイ再注入 (ウォームアップ戻し時等)
            if self.shared_replay is not None:
                ag.replay_buffer = self.shared_replay
        # 既存リプレイのロード (継続学習対応)
        if self.shared_replay is not None:
            replay_path = self.config.get("replay_path", "replay_buffer.joblib")
            if os.path.exists(replay_path):
                try:
                    loaded = ReplayBuffer.load(replay_path)
                    # load は新インスタンスを返すので差し替え
                    self.shared_replay = loaded
                    for ag in self.agents:
                        if isinstance(ag, AlphaZeroAgent):
                            ag.replay_buffer = self.shared_replay
                    if self.logger:
                        self.logger.log_text(f"[replay] loaded existing shared buffer size={len(self.shared_replay)}")
                    else:
                        print(f"[INFO] loaded replay size={len(self.shared_replay)}")
                except Exception as e:
                    print(f"[WARN] replay load failed: {e}")
        # 初期チェックポイント (空のリプレイと初期モデル) を要求された場合に保存
        # ただし、既存 ckpt からの再開時は二重保存を避けるためスキップ
        if self.config.get("initial_checkpoint_on_setup", False) and (not getattr(self, "_loaded_from_checkpoint", False)):
            try:
                self._save_checkpoint(version_tag=None)
                if self.logger:
                    self.logger.log_text("[init] initial checkpoint saved")
            except Exception as e:
                print(f"[WARN] initial checkpoint save failed: {e}")

    # ウォームアップ用: 他プレイヤーを簡易エージェントに差し替え
    def _apply_warmup_opponents(self):
        from agents.random_agent import RandomAgent
        from agents.rule_based_agent import RuleBasedAgent
        new_list = []
        for i in range(self.config["num_players"]):
            if i == self.learning_player_id:
                # 学習対象プレイヤーは AlphaZero のまま
                new_list.append(self.agents[i])
            else:
                spec = None
                if self.warmup_mix:
                    spec = self.warmup_mix[(i - (1 if i > self.learning_player_id else 0)) % len(self.warmup_mix)]
                if spec == "rule":
                    new_list.append(RuleBasedAgent(player_id=i))
                else:
                    new_list.append(RandomAgent(player_id=i))
        # トレーナ保持リスト置換
        self.agents = new_list
        # 環境内の agents も同期
        self.env.agents = self.agents
        for ag in self.agents:
            if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'set_env_ref'):
                ag.set_env_ref(self.env)
                if self.shared_replay is not None:
                    ag.replay_buffer = self.shared_replay

    def _restore_learning_agents(self):
        # すべて AlphaZeroAgent に戻す (学習対象以外を新規インスタンスにしてもよいが、簡単のため再生成)
        if not any(isinstance(a, AlphaZeroAgent) for a in self.agents if a.player_id == self.learning_player_id):
            # 念のため学習プレイヤーが消えていたら復元
            self.agents[self.learning_player_id] = AlphaZeroAgent(player_id=self.learning_player_id, model=self.model, config=self.config)
        for i in range(self.config["num_players"]):
            if i == self.learning_player_id:
                continue
            if not isinstance(self.agents[i], AlphaZeroAgent):
                self.agents[i] = AlphaZeroAgent(player_id=i, model=self.model, config=self.config)
        self.env.agents = self.agents
        for ag in self.agents:
            if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'set_env_ref'):
                ag.set_env_ref(self.env)
                if self.shared_replay is not None:
                    ag.replay_buffer = self.shared_replay

    # -----------------------------------------------------
    # High/Low Water 自動リプレイ縮小 (メモリベース専用)
    # -----------------------------------------------------
    def _auto_replay_water_purge(self):
        """メモリ(RSS)ベースの自動リプレイ縮小 (親+子プロセス合算対応)。"""
        cfg = self.config
        if cfg.get('purge_replay_after_each_update') or not cfg.get('auto_replay_water_enabled', False):
            return
        rb = self.shared_replay or (self.agents and getattr(self.agents[0], 'replay_buffer', None))
        if rb is None:
            return
        try:
            capacity = int(cfg.get('buffer_size', 1))
            cur = len(rb) if hasattr(rb, '__len__') else None
        except Exception:
            return
        if not cur or capacity <= 0:
            return
        try:
            import psutil  # type: ignore
            proc = psutil.Process()
        except Exception:
            return
        include_children = bool(cfg.get('replay_memory_include_children', False))
        child_interval = int(cfg.get('replay_memory_children_recalc_sec', 15) or 15)
        now = time.time()
        try:
            parent_rss = proc.memory_info().rss
            total_mem = psutil.virtual_memory().total
        except Exception:
            return
        children_rss = 0
        if include_children:
            last_children_ts = getattr(self, '_last_children_rss_ts', 0.0)
            cached_children = getattr(self, '_last_children_rss', None)
            if (now - last_children_ts) < child_interval and cached_children is not None:
                children_rss = cached_children
            else:
                try:
                    total_ch = 0
                    child_infos = []
                    for c in proc.children(recursive=True):
                        try:
                            if not c.is_running():
                                continue
                            mi = c.memory_info().rss
                            total_ch += mi
                            child_infos.append((c.pid, mi, getattr(c, 'name', lambda: '')()))
                        except Exception:
                            continue
                    # 上位 N = 5 をログ (debug でなくても高水位診断のため常時発火時のみ)
                    child_infos.sort(key=lambda x: x[1], reverse=True)
                    top_n = child_infos[:5]
                    if top_n and self.logger:
                        try:
                            line = ", ".join([f"pid={pid} rss={rss/(1024**3):.2f}GB" for pid, rss, _nm in top_n])
                            # 子プロセスRSS概要ログ (ノイズ低減のためデフォルト無効化可能)
                            if bool(self.config.get('replay_log_child_rss', False)):
                                self.logger.log_text(f"[replay] child_rss_top total_children={len(child_infos)} top5=[{line}]")
                        except Exception:
                            pass
                    children_rss = total_ch
                    self._last_children_rss = children_rss
                    self._last_children_rss_ts = now
                except Exception:
                    children_rss = 0
        rss = parent_rss + children_rss
        total = total_mem
        high_ratio = float(cfg.get('replay_memory_high_ratio', 0.8) or 0.8)
        low_ratio = float(cfg.get('replay_memory_low_ratio', 0.6) or 0.6)
        abs_mb = int(cfg.get('replay_memory_high_abs_mb', 0) or 0)
        low_abs_mb = int(cfg.get('replay_memory_low_abs_mb', 0) or 0)
        cooldown = int(cfg.get('replay_memory_cooldown_sec', 300) or 300)
        emergency_ratio = float(cfg.get('replay_memory_emergency_ratio', 0.0) or 0.0)
        aggressive_factor = float(cfg.get('replay_memory_aggressive_factor', 0.8) or 0.8)
        min_purge_rows = int(cfg.get('replay_memory_min_purge_rows', 0) or 0)
        debug_log = bool(cfg.get('replay_memory_debug_log', False))
        log_before_after = bool(cfg.get('replay_memory_log_before_after', False))
        force_gc = bool(cfg.get('replay_memory_force_gc', False))
        compact_mode = cfg.get('replay_memory_compact_mode', 'none') or 'none'
        if not (0.0 < low_ratio < high_ratio < 1.0):
            return
        usage_ratio = rss / total if total else 0.0
        over_ratio = usage_ratio >= high_ratio
        over_abs = abs_mb > 0 and (rss >= abs_mb * 1024 * 1024)
        if debug_log:
            msg_chk = (f"[replay] mem_check ratio={usage_ratio:.4f} parent={parent_rss/(1024**3):.2f}GB "
                       f"children={children_rss/(1024**3):.2f}GB total={rss/(1024**3):.2f}GB high={high_ratio:.2f} "
                       f"low={low_ratio:.2f} over_ratio={over_ratio} over_abs={over_abs}")
            if self.logger:
                try: self.logger.log_text(msg_chk)
                except Exception: pass
            else:
                print(msg_chk)
        if not (over_ratio or over_abs):
            return
        last = getattr(self, '_last_mem_water_ts', 0.0)
        if (now - last) < cooldown:
            return
        cur_usage_ratio = usage_ratio
        # --- ターゲットサイズ計算 ---
        # 通常: ratio を low_ratio まで下げることを目標に計算
        target_size = int(capacity * (cur / capacity * (low_ratio / max(cur_usage_ratio, 1e-9))))
        min_target = int(capacity * low_ratio)
        if target_size < min_target:
            target_size = min_target
        # 絶対高水位 (abs_mb) で発火した & low_abs_mb (>0) 指定時は絶対メモリを基準に再計算
        if over_abs and low_abs_mb > 0:
            # 期待メモリ削減率を (low_abs_mb / abs_mb) としてサンプル数へ反映 (単純比例仮定)
            ratio_factor = low_abs_mb / max(abs_mb, 1)
            alt_target = int(cur * ratio_factor)
            # low_ratio に基づく min_target と比較し、より小さい方を採用 (メモリ圧縮優先)
            if alt_target < target_size:
                target_size = max(0, alt_target)
        if emergency_ratio and usage_ratio >= emergency_ratio and 0.0 < aggressive_factor < 1.0:
            target_size = int(target_size * aggressive_factor)
        if min_purge_rows > 0 and (cur - target_size) < min_purge_rows:
            target_size = max(0, cur - min_purge_rows)
        if target_size < 0:
            target_size = 0
        if target_size >= cur:
            self._last_mem_water_ts = now
            return
        # -----------------------------
        # 削除予定サンプルのみ追記形式で joblib に蓄積
        # -----------------------------
        remove_count = cur - target_size
        if remove_count > 0:
            try:
                snapshot_dir = self.config.get('checkpoint_dir', 'checkpoints')
                os.makedirs(snapshot_dir, exist_ok=True)
                append_path = os.path.join(snapshot_dir, 'replay_autopurge_append.joblib')
                oldest_list = []
                # 可能ならロック下で先頭要素をコピー
                from itertools import islice
                if hasattr(rb, '_data'):
                    lock = getattr(rb, '_lock', None)
                    if lock is not None:
                        with lock:  # type: ignore
                            oldest_list = list(islice(rb._data, 0, remove_count))  # type: ignore[attr-defined]
                    else:
                        oldest_list = list(islice(rb._data, 0, remove_count))  # type: ignore[attr-defined]
                elif hasattr(rb, '__iter__'):
                    # フォールバック: iter から取得 (順序保証されない可能性)
                    try:
                        it = iter(rb)
                        for _ in range(remove_count):
                            try:
                                oldest_list.append(next(it))
                            except StopIteration:
                                break
                    except Exception:
                        oldest_list = []
                # 最低限のキーだけ抽出してサイズ抑制 (ReplayBuffer.save の allow_keys と揃える)
                allow_keys = {"player_id","state","value","model_version","feature_version","value_pred","uid","pi_q","pi_format","legal_ids","actions_format","value_u8","value_pred_u8"}
                processed = []
                for s in oldest_list:
                    if not isinstance(s, dict):
                        continue
                    try:
                        d = {k: s.get(k) for k in allow_keys if k in s}
                        d['autopurge_ts'] = time.time()
                        processed.append(d)
                    except Exception:
                        continue
                import joblib
                # 既存ファイルがあれば読み込み拡張 (単純 append モデル)
                if os.path.exists(append_path):
                    try:
                        obj = joblib.load(append_path)
                        if isinstance(obj, dict):
                            base_list = obj.get('data', [])
                        elif isinstance(obj, list):
                            base_list = obj
                        else:
                            base_list = []
                    except Exception:
                        base_list = []
                else:
                    base_list = []
                base_list.extend(processed)
                # オプション: 過剰肥大化防止 (例: 200万件超で後ろ 150万件にトリム) ※固定値簡易実装
                if len(base_list) > 2_000_000:
                    base_list = base_list[-1_500_000:]
                # ReplayBuffer.save と同形式: {maxlen, next_id, data}
                next_id_val = None
                try:
                    next_id_val = getattr(rb, '_next_id', None)
                except Exception:
                    next_id_val = None
                payload = {'maxlen': getattr(rb, 'maxlen', None), 'next_id': next_id_val, 'data': base_list}
                # 原子的保存 (tmp -> replace)
                tmp_ap = append_path + '.tmp'
                try:
                    joblib.dump(payload, tmp_ap, compress=3)
                    os.replace(tmp_ap, append_path)
                except Exception:
                    try:
                        if os.path.exists(tmp_ap):
                            os.remove(tmp_ap)
                    except Exception:
                        pass
                    # フォールバック直接保存
                    try:
                        joblib.dump(payload, append_path, compress=0)
                    except Exception:
                        pass
                if self.logger:
                    try:
                        self.logger.log_text(f"[replay] autopurge_append_saved path={append_path} added={len(processed)} total={len(base_list)} remove_count={remove_count} target={target_size}")
                    except Exception:
                        pass
            except Exception:
                pass
        if hasattr(rb, 'shrink_to_size'):
            removed = rb.shrink_to_size(target_size)
            new_size = len(rb)
        else:
            removed = 0
            over = cur - target_size
            try:
                if over > 0 and hasattr(rb, 'popleft'):
                    for _ in range(over):
                        try: rb.popleft(); removed += 1
                        except Exception: break
                elif over > 0 and isinstance(rb, list):
                    del rb[:over]; removed = over
                new_size = len(rb)
            except Exception:
                return
        if compact_mode == 'rebuild' and hasattr(rb, '_data'):
            try:
                from collections import deque as _dq
                if hasattr(rb, '_lock'):
                    with rb._lock:  # type: ignore[attr-defined]
                        data_list = list(rb._data)  # type: ignore[attr-defined]
                        rb._data = _dq(data_list, maxlen=rb.maxlen)  # type: ignore[attr-defined]
                else:
                    data_list = list(rb._data)  # type: ignore[attr-defined]
                    rb._data = _dq(data_list, maxlen=rb.maxlen)  # type: ignore[attr-defined]
            except Exception:
                pass
        after_delete_rss = rss
        if log_before_after and self.logger:
            try: self.logger.log_text(f"[replay] memory_purge after_delete removed={removed} cur={new_size} target={target_size} ratio_before={cur_usage_ratio:.4f}")
            except Exception: pass
        if force_gc:
            try:
                import gc; gc.collect()
                parent_after = proc.memory_info().rss
                children_after = 0
                if include_children:
                    try:
                        for c in proc.children(recursive=True):
                            try:
                                if c.is_running(): children_after += c.memory_info().rss
                            except Exception: continue
                    except Exception: pass
                total_after = parent_after + children_after
                freed = (after_delete_rss - total_after) / (1024**3)
                if log_before_after:
                    msg_gc = (f"[replay] memory_purge after_gc total={total_after/(1024**3):.2f}GB freed≈{freed:.2f}GB")
                    if self.logger:
                        try: self.logger.log_text(msg_gc)
                        except Exception: pass
                    else: print(msg_gc)
            except Exception:
                pass
        self._last_mem_water_ts = now
        if removed > 0:
            summary = (f"[replay] auto_memory_purge ratio={cur_usage_ratio:.2f} removed={removed} cur={new_size} "
                       f"cap={capacity} parent={parent_rss/(1024**3):.2f}GB children={children_rss/(1024**3):.2f}GB high={high_ratio:.2f} low={low_ratio:.2f}")
            if self.logger:
                try: self.logger.log_text(summary)
                except Exception: pass
            else:
                print(summary)

    # -----------------------------------------------------
    # 自己対局 (データ収集)
    # -----------------------------------------------------
    def self_play(self, num_episodes: int = 1):
        # 並列数で分岐
        workers = int(self.config.get("selfplay_workers", 0) or 0)
        if workers and workers > 1:
            return self._self_play_parallel(num_episodes=num_episodes, workers=workers)

        start_time = time.time()
        # 実行開始直後に 0% 進捗を表示して無音時間を減らす
        if self.minimal_progress and self.use_progress_bar and num_episodes > 0:
            bar, _pct = self._make_progress_bar(0, num_episodes)
            line = f"[SELFPLAY] {bar} 0/{num_episodes}"
            print(line, end='\r', flush=True)
            self._last_progress_len = len(line)
        for ep in range(num_episodes):
            ep_start = time.time()
            if not self.minimal_progress and not self.use_progress_bar:
                print(f"[EPOCH] {ep+1}/{num_episodes}")
            # ウォームアップ期間中は対戦相手をランダム/ルールベースに
            # 複数回の self_play 呼び出しでも通算エピソード数で判定
            if self._episodes_total_run < self.warmup_episodes:
                self._apply_warmup_opponents()
            elif self._episodes_total_run == self.warmup_episodes:
                # ウォームアップ直後に全員学習エージェントへ戻す
                self._restore_learning_agents()
            self._play_one_episode(ep)
            self._episodes_total_run += 1
            # 周期チェックポイント保存 (2000 エピソードごと等)
            if self.ckpt_interval > 0 and (self._episodes_total_run % self.ckpt_interval == 0):
                self._save_checkpoint(version_tag=f"ep{self._episodes_total_run}")
                # --- 過去モデルプールへ追加 (多世代保持) ---
                if self.keep_prev_model:
                    self._snapshot_current_model()
                # --- 対戦相手へ過去モデル再割当 (プールが空なら従来方式にフォールバック) ---
                if self.keep_prev_model and self.prev_model_mix_players > 0:
                    if self.past_models:
                        self._assign_past_models_to_opponents()
                    elif self._previous_model is not None:  # 後方互換: 旧単一方式
                        self._mix_previous_model_opponents()
                # チェックポイント間隔で世代を進め、エージェントへ新世代番号を反映
                self.model_version += 1
                for ag in self.agents:
                    if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'model_version'):
                        ag.model_version = self.model_version
            # 任意のエピソード間隔で opponent 再割当 (checkpoint タイミング以外でも多様性確保)
            if self.opponent_mix_interval > 0 and self.keep_prev_model and self.prev_model_mix_players > 0:
                if (self._episodes_total_run % self.opponent_mix_interval == 0) and self.past_models:
                    self._assign_past_models_to_opponents()

            # High/Low water purge (single-process self_play)
            try:
                self._auto_replay_water_purge()
            except Exception:
                pass

            # 進捗表示 (エピソード終了後に確定時間で ETA 推定)
            if self.minimal_progress and self.use_progress_bar:
                done = ep + 1
                ep_dur = time.time() - ep_start
                # エピソード時間 EMA
                if self._eta_smooth is None:
                    self._eta_smooth = ep_dur
                else:
                    a = max(0.0, min(1.0, self.eta_alpha))
                    self._eta_smooth = a * ep_dur + (1 - a) * self._eta_smooth
                elapsed = time.time() - start_time
                estimated_total = (self._eta_smooth or 0.0) * num_episodes
                remaining = max(0.0, estimated_total - elapsed)
                if self.monotonic_eta and self._eta_prev_remaining is not None and remaining > self._eta_prev_remaining:
                    remaining = self._eta_prev_remaining
                self._eta_prev_remaining = remaining
                def _fmt(t: float):
                    m, s = divmod(int(t), 60)
                    h, m = divmod(m, 60)
                    return f"{h:d}:{m:02d}:{s:02d}"
                bar, _pct = self._make_progress_bar(done, num_episodes)
                line = f"[SELFPLAY] {bar} {done}/{num_episodes} eta={_fmt(remaining)}"
                pad = max(0, self._last_progress_len - len(line))
                print(line + ' ' * pad, end='\r' if done < num_episodes else '\n', flush=True)
                self._last_progress_len = len(line)
        # ループ終了後、バー表示時は改行が確定するので追加処理不要
        # csv_summary_only 用最終1行書き出し (self_play 単体利用ケースでも確実にファイル生成)
        try:
            if self.logger and getattr(self.logger, 'csv_summary_only', False):
                self.logger.write_csv_summaries()
        except Exception as e:
            print(f"[WARN] write_csv_summaries(self_play) failed: {e}")
        return

    # -----------------------------------------------------
    # 並行実行: 常駐自己対局(Producer) + 親プロセス学習(Consumer)
    # -----------------------------------------------------
    def train_concurrent(
        self,
        *,
        total_episodes: int,
        workers: int | None = None,
        updates_per_iter: int = 50,
        queue_maxsize: int = 15000,
        progress_print_every: int = 50,
    ):
        """自己対局をワーカープロセスで常時生成しつつ、親で学習を並行実行。

        仕様:
          - ワーカーはサンプルを sample_queue に push、親は随時消費して train_updates を小刻みに回す。
          - 既存のチェックポイント挙動を維持:
              * エピソード間隔 ckpt: ワーカーからの "ep_done" を集計し、ckpt_interval 到達時に保存
              * 学習後 ckpt: 親の学習イテレーションの各バッチグループ後に _save_checkpoint() を呼ぶ
          - 終了条件: total_episodes に達した時点でワーカー停止、残キューを消費して終了
        """
        assert total_episodes > 0
        workers = int(workers if workers is not None else (self.config.get("selfplay_workers", 0) or 0))
        workers = max(1, workers)

        # モデル配布ファイル: 常に最新チェックポイントを参照させる
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        model_blob_path = self.config.get("checkpoint_path", os.path.join(self.config["checkpoint_dir"], "policy_value_latest.pt"))
        # 初回: latest が未作成なら現モデルを保存
        if (self.model is not None) and (not os.path.exists(model_blob_path)):
            try:
                self.model.save(model_blob_path)
            except Exception:
                pass

        # 共有リプレイの存在に応じて取り込み先を決定
        dst_buffer = self.shared_replay if self.shared_replay is not None else None
        learner = self.agents[self.learning_player_id]
        if dst_buffer is None and isinstance(learner, AlphaZeroAgent):
            dst_buffer = learner.replay_buffer

        # 並行用 Queue / Event
        ctx = mp.get_context("spawn")
        sample_queue = ctx.Queue(maxsize=max(1000, queue_maxsize))
        event_queue = ctx.Queue(maxsize=10000)
        stop_event = ctx.Event()
        # 再起動や停止時のグレースフルフラッシュ制御専用キュー (各 worker 用)
        control_queues: List[Any] = []

        # ワーカー起動
        procs: List[mp.Process] = []
        for wid in range(workers):
            cq = ctx.Queue(maxsize=5)
            control_queues.append(cq)
            p = ctx.Process(
                target=_selfplay_daemon_worker,
                args=(wid, self.config, model_blob_path, sample_queue, event_queue, stop_event, cq),
                daemon=True,
            )
            p.start()
            procs.append(p)
        # 起動確認ログは簡潔化のため削除

        # 進捗表示初期化
        if self.minimal_progress and self.use_progress_bar:
            bar, _pct = self._make_progress_bar(0, total_episodes)
            line = f"[SELFPLAY~] {bar} 0/{total_episodes} (workers={workers})"
            print(line, end='\r', flush=True)
            last_len_sp = len(line)
        else:
            last_len_sp = 0

        # ループ: イベントを処理しつつ、溜まったサンプルで学習を回す
        ep_done = 0
        train_it = 0
        ckpt_interval = int(self.ckpt_interval or 0)
        # サンプル数で学習を発火
        min_new_samples_before_train = int(self.config.get("concurrent_min_new_samples_before_train", 2000) or 2000)
        new_samples_since_train = 0
        # 保存周りを秒ベースでスロットリングして I/O 負荷を軽減
        latest_ckpt_interval_sec = float(self.config.get("concurrent_latest_save_every_sec", 30.0) or 0.0)
        blob_save_interval_sec = float(self.config.get("concurrent_blob_save_every_sec", 30.0) or 0.0)
        last_latest_ckpt_ts = 0.0
        last_blob_save_ts = 0.0
        # 親側で episodes_total_run を進めるため、event_queue からの "ep_done" をカウント
        status_log_sec = float(self.config.get("concurrent_status_log_sec", 0) or 0)
        status_log_include_mem = bool(self.config.get("status_log_include_memory", False))
        debug_flag = bool(self.config.get("concurrent_debug_logging", False))
        last_status_ts = time.time()

        def _drain_samples(max_items: int | None = None):
            nonlocal dst_buffer
            consumed = 0
            skipped = 0
            import random as _r
            while True:
                if max_items is not None and consumed >= max_items:
                    break
                try:
                    s = sample_queue.get_nowait()
                except Exception:
                    break
                # サンプルバリデーション
                if not isinstance(s, dict):
                    skipped += 1
                    continue
                # 量子化 pi_q のみを保持する新形式にも対応
                if 'state' not in s or not ('pi' in s or 'pi_q' in s):
                    skipped += 1
                    continue
                # 取り込み
                # 学習/検証スプリットが未設定ならここで付与（ワーカーで未設定の場合の後方互換）
                try:
                    if s.get('split') is None:
                        ratio = float(self.config.get('val_split_ratio', 0.0) or 0.0)
                        s['split'] = 'val' if (_r.random() < ratio) else 'train'
                except Exception:
                    pass
                if self.shared_replay is not None:
                    try:
                        self.shared_replay.append(s)
                    except Exception:
                        pass
                else:
                    try:
                        dst_buffer.append(s)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                consumed += 1
            if debug_flag and skipped > 0:
                print(f"[DEBUG] drain skipped={skipped} accepted={consumed} (reason: missing state or pi/pi_q)")
            return consumed

        # ---------------- Worker Restart (Minimal Implementation) ----------------
        # 内部状態テーブル: per worker
        restart_cfg = {
            'enable': bool(self.config.get('worker_restart_enable', False)),
            'high_mb': int(self.config.get('worker_restart_rss_high_mb', 0) or 0),
            'consecutive': int(self.config.get('worker_restart_consecutive_required', 2) or 2),
            'min_interval': int(self.config.get('worker_restart_min_interval_sec', 600) or 600),
            'jitter': int(self.config.get('worker_restart_jitter_sec', 0) or 0),
            'emergency_total_mb': int(self.config.get('worker_restart_emergency_total_mb', 0) or 0),
            'grace_timeout': int(self.config.get('worker_restart_grace_timeout_sec', 120) or 120),
            'force_kill_sec': int(self.config.get('worker_restart_force_kill_sec', 150) or 150),
            'flush_timeout': int(self.config.get('worker_restart_flush_timeout_sec', 20) or 20),
            'log_obj_on_exit': bool(self.config.get('worker_restart_log_object_types_on_exit', True)),
        }
        worker_stats = {}
        for idx, p in enumerate(procs):  # 初期化
            worker_stats[idx] = {
                'rss_high_count': 0,
                'last_restart': 0.0,
                'generation': 0,
                'pending': False,
                'grace_start': None,
            }

        def _seed_for(worker_id: int, generation: int):
            base = int(self.config.get('seed', 42))
            return base + 10000 * worker_id + 500 * generation

        def _psutil_process(pid):
            try:
                import psutil  # type: ignore
                return psutil.Process(pid)
            except Exception:
                return None

        def _check_and_mark_restarts():
            # 緊急しきい値が設定されている場合は、enable/high_mb に関わらずチェックを行う。
            # 通常の高水位判定は enable 且つ high_mb>0 の場合のみ有効化。
            has_emergency = int(restart_cfg.get('emergency_total_mb', 0) or 0) > 0
            has_normal_high = restart_cfg['enable'] and (restart_cfg['high_mb'] > 0)
            if not has_emergency and not has_normal_high:
                # どちらのモードも有効でない場合は即 return
                return []
            marked = []
            try:
                import psutil  # type: ignore
            except Exception:
                return []
            # 親 + 子(ワーカー + 評価ゲート等)の RSS 合算を算出
            parent_rss = 0
            try:
                parent_rss = psutil.Process().memory_info().rss
            except Exception:
                parent_rss = 0
            total_rss_children = 0
            rss_by_wid = {}
            for p in procs:
                if not p.is_alive():
                    continue
                proc_obj = _psutil_process(p.pid)
                if proc_obj is None:
                    continue
                try:
                    rss = proc_obj.memory_info().rss
                    total_rss_children += rss
                    rss_by_wid[p.pid] = rss
                except Exception:
                    continue
            # 評価ゲートなどワーカー以外の子プロセスも合算に含める（現状はゲートのみ明示対応）
            gate_rss = 0
            gate_pid = None
            try:
                gp = getattr(self, "_gate_proc", None)
                if gp is not None and gp.is_alive():
                    proc_gate = _psutil_process(gp.pid)
                    if proc_gate is not None:
                        gate_rss = proc_gate.memory_info().rss
                        gate_pid = gp.pid
                        total_rss_children += gate_rss
            except Exception:
                pass
            total_rss = parent_rss + total_rss_children
            emergency_threshold = restart_cfg['emergency_total_mb'] * 1024 * 1024 if restart_cfg['emergency_total_mb'] > 0 else 0
            emergency = emergency_threshold > 0 and (total_rss >= emergency_threshold)

            high_list = []
            # 通常の高水位しきい値チェックは、設定が有効なときのみ評価
            if has_normal_high:
                for wid, p in enumerate(procs):
                    if not p.is_alive():
                        continue
                    proc_obj = _psutil_process(p.pid)
                    if proc_obj is None:
                        continue
                    try:
                        rss = proc_obj.memory_info().rss
                    except Exception:
                        continue
                    rss_mb = rss / (1024 * 1024)
                    st = worker_stats.get(wid)
                    if st is None:
                        continue
                    if rss_mb >= restart_cfg['high_mb']:
                        st['rss_high_count'] += 1
                    else:
                        st['rss_high_count'] = 0
                    eligible = (st['rss_high_count'] >= restart_cfg['consecutive'] and
                                (time.time() - st['last_restart']) >= restart_cfg['min_interval'] and not st['pending'])
                    if eligible:
                        high_list.append((wid, rss_mb))
            # 緊急: 全体RSSがしきい値超過なら、個別閾値に関わらず最大RSSのプロセスを1件対象
            target_wids = []
            if emergency:
                # 候補: ワーカー（min_interval を緩やかに尊重）とゲートプロセス
                candidates = []  # (kind, id_or_pid, rss, ok_interval)
                for wid, p in enumerate(procs):
                    if not p.is_alive():
                        continue
                    st = worker_stats.get(wid)
                    if st is None or st.get('pending'):
                        continue
                    # RSS 取得
                    proc_obj = _psutil_process(p.pid)
                    if proc_obj is None:
                        continue
                    try:
                        rss_now = proc_obj.memory_info().rss
                    except Exception:
                        continue
                    # min_interval を緩やかに尊重（緊急時でも直後の連続再起動は避ける）
                    since = time.time() - st.get('last_restart', 0.0)
                    ok_interval = since >= restart_cfg['min_interval']
                    candidates.append(("worker", wid, rss_now, ok_interval))
                # ゲートプロセスを候補に含める（min_interval の概念はないため常に ok）
                if gate_pid is not None and gate_rss > 0:
                    candidates.append(("gate", gate_pid, gate_rss, True))
                if candidates:
                    # まず min_interval を満たす中で最大RSS を選択、なければ全体の最大RSS
                    c_ok = [c for c in candidates if c[3]]
                    pick_from = c_ok if c_ok else candidates
                    kind_pick, ident_pick, rss_pick, _ = max(pick_from, key=lambda x: x[2])
                    if kind_pick == "worker":
                        target_wids = [int(ident_pick)]
                        if self.logger:
                            try:
                                self.logger.log_text(
                                    f"[worker-restart] emergency trigger total_rss={total_rss/(1024**3):.2f}GB parent={parent_rss/(1024**3):.2f}GB threshold={restart_cfg['emergency_total_mb']/1024:.2f}GB wid={int(ident_pick)}"
                                )
                            except Exception:
                                pass
                    elif kind_pick == "gate":
                        # 評価ゲートプロセスを終了させる（安全側: terminate）
                        try:
                            gp = getattr(self, "_gate_proc", None)
                            if gp is not None and gp.is_alive():
                                gp.terminate()
                                try:
                                    gp.join(timeout=3)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        try:
                            # ハンドルをクリア
                            self._gate_proc = None
                            self._gate_queue = None
                            self._gate_candidate_path = None
                            self._gate_baseline_path = None
                        except Exception:
                            pass
                        if self.logger:
                            try:
                                self.logger.log_text(
                                    f"[worker-restart] emergency trigger total_rss={total_rss/(1024**3):.2f}GB parent={parent_rss/(1024**3):.2f}GB threshold={restart_cfg['emergency_total_mb']/1024:.2f}GB gate_pid={int(ident_pick)} terminated"
                                )
                            except Exception:
                                pass
            else:
                # 通常: 個別高水位のものをすべて対象に
                target_wids = [wid for wid, _ in high_list]
            for wid in target_wids:
                st = worker_stats[wid]
                st['pending'] = True
                st['grace_start'] = time.time()
                marked.append(wid)
                if self.logger:
                    try:
                        self.logger.log_text(f"[worker-restart] mark wid={wid} gen={st['generation']} reason={'emergency' if emergency else 'high_rss'}")
                    except Exception:
                        pass
            return marked

        def _issue_graceful_restart(wid_list):
            # 改良版: まず FLUSH_AND_EXIT 制御を送信し、一定時間待機 → 未終了なら terminate。
            for wid in wid_list:
                p = procs[wid]
                st = worker_stats[wid]
                if not p.is_alive():
                    continue
                # 1) flush 指示
                try:
                    control_queues[wid].put('FLUSH_AND_EXIT', block=False)
                except Exception:
                    pass
            # 2) 待機 & shard 取り込み (後段で _merge_shards を呼ぶ)
            deadline = time.time() + restart_cfg['grace_timeout']
            still_alive = set(wid_list)
            while still_alive and time.time() < deadline:
                for wid in list(still_alive):
                    p = procs[wid]
                    if not p.is_alive():
                        still_alive.discard(wid)
                time.sleep(0.1)
            # 3) まだ生きているものを強制終了
            for wid in list(still_alive):
                p = procs[wid]
                try:
                    p.terminate()
                except Exception:
                    pass
                try:
                    p.join(timeout=3)
                except Exception:
                    pass
            # 4) 再spawn
            for wid in wid_list:
                p_old = procs[wid]
                if p_old.is_alive():
                    # 既に再利用 (強制 kill 失敗) はスキップ
                    continue
                new_cq = mp.get_context('spawn').Queue(maxsize=5)
                control_queues[wid] = new_cq
                new_p = mp.get_context('spawn').Process(
                    target=_selfplay_daemon_worker,
                    args=(wid, self.config, model_blob_path, sample_queue, event_queue, stop_event, new_cq),
                    daemon=True,
                )
                new_p.start()
                procs[wid] = new_p
                st = worker_stats[wid]
                st['generation'] += 1
                st['last_restart'] = time.time()
                st['pending'] = False
                st['rss_high_count'] = 0
                if self.logger:
                    try:
                        self.logger.log_text(f"[worker-restart] done wid={wid} gen={st['generation']} new_pid={new_p.pid}")
                    except Exception:
                        pass
            # 5) 再起動後にシャードをマージ
            _merge_shards()

        def _merge_shards():
            """replay_shards ディレクトリの未マージファイルを読み込み central buffer へ統合。"""
            shard_dir = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'replay_shards')
            if not os.path.isdir(shard_dir):
                return
            merged_dir = os.path.join(shard_dir, 'merged')
            os.makedirs(merged_dir, exist_ok=True)
            try:
                files = [f for f in os.listdir(shard_dir) if f.endswith('.joblib') and os.path.isfile(os.path.join(shard_dir, f))]
            except Exception:
                return
            if not files:
                return
            imported = 0
            for fn in files:
                fp = os.path.join(shard_dir, fn)
                try:
                    import joblib as _jb
                    data = _jb.load(fp)
                    if isinstance(data, list):
                        for s in data:
                            if not isinstance(s, dict):
                                continue
                            if self.shared_replay is not None:
                                try: self.shared_replay.append(s)
                                except Exception: pass
                            else:
                                try: dst_buffer.append(s)  # type: ignore[attr-defined]
                                except Exception: pass
                            imported += 1
                    # 移動
                    try:
                        os.replace(fp, os.path.join(merged_dir, fn))
                    except Exception:
                        pass
                except Exception:
                    # 読み込み失敗はスキップ (残して次回再試行)
                    continue
            if imported and self.logger:
                try:
                    self.logger.log_text(f"[shard-merge] imported={imported} files={len(files)}")
                except Exception:
                    pass

        last_worker_rss_check = 0.0
        worker_rss_check_interval = 15.0  # 秒 (最小安全版固定)

        # ループ開始時の詳細ログは削除（簡潔化）

        try:
            while ep_done < total_episodes:
                # イベント処理（自己対局進捗）
                try:
                    evt, val = event_queue.get(timeout=0.1)
                except Exception:
                    evt = None
                if evt == "ep_done":
                    ep_done += int(val)
                    self._episodes_total_run += int(val)
                    # --------------------------------------------------
                    # エピソード間隔チェックポイント
                    # 以前は obj_types イベント(50ep毎) 側でのみ発火していたため、
                    # ckpt_interval が obj_types 発火周期と一致しない場合に取りこぼしが発生。
                    # (例: ckpt_interval=100 かつ obj_types間隔=50 の場合は 100,200 はOK だが
                    #      120 など 50 の倍数でない値を設定すると一度も一致しない可能性)
                    # ep_done 受信直後に判定へ移し、正確に interval ごと保存するよう修正。
                    # --------------------------------------------------
                    if ckpt_interval > 0 and (self._episodes_total_run % ckpt_interval == 0):
                        try:
                            self._save_checkpoint(version_tag=f"ep{self._episodes_total_run}")
                            if self.keep_prev_model:
                                self._snapshot_current_model()
                            if self.keep_prev_model and self.prev_model_mix_players > 0:
                                if self.past_models:
                                    self._assign_past_models_to_opponents()
                                elif self._previous_model is not None:
                                    self._mix_previous_model_opponents()
                            # モデル世代反映
                            self.model_version += 1
                            for ag in self.agents:
                                if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'model_version'):
                                    ag.model_version = self.model_version
                        except Exception:
                            pass
                elif evt == "play_stats":
                    # ワーカーからのプレイ統計: payload dict
                    try:
                        st = val or {}
                        wid = st.get('wid')
                        w_ep = st.get('ep')
                        def _fmt(x):
                            return f"{x:.2f}" if isinstance(x, (int, float)) and x is not None else (str(x) if x is not None else 'n/a')
                        line = (
                            f"[playstats] wid={wid} ep={w_ep} "
                            f"moves_avg={_fmt(st.get('moves_avg'))} moves_p95={_fmt(st.get('moves_p95'))} "
                            f"sims_avg={_fmt(st.get('sims_avg'))} sims_p50={_fmt(st.get('sims_p50'))} sims_p95={_fmt(st.get('sims_p95'))} "
                            f"early_stop_rate={_fmt(st.get('early_stop_rate'))} n={int(st.get('sample_size') or 0)}"
                        )
                        if self.logger:
                            self.logger.log_text(line)
                        else:
                            print(line)
                    except Exception:
                        pass

                    # 進捗表示
                    if self.minimal_progress and self.use_progress_bar:
                        bar, _pct = self._make_progress_bar(ep_done, total_episodes)
                        l2 = f"[SELFPLAY~] {bar} {ep_done}/{total_episodes} (workers={workers})"
                        pad = max(0, last_len_sp - len(l2))
                        print(l2 + ' ' * pad, end='\r' if ep_done < total_episodes else '\n', flush=True)
                        last_len_sp = len(l2)
                elif evt == "hb":
                    # ワーカーハートビート: 長いエピソードでも親から可視化
                    try:
                        wid, steps = val
                        if self.logger:
                            self.logger.log_text(f"[worker-hb] wid={wid} steps={steps}")
                    except Exception:
                        pass

                # サンプル取り込み（少しずつ）
                consumed_now = _drain_samples(max_items=500)
                new_samples_since_train += int(consumed_now)
                # 取り込み実績の詳細ログは簡潔化のため削除
                # High/Low water 自動 purge
                if consumed_now > 0:
                    try:
                        self._auto_replay_water_purge()
                    except Exception:
                        pass
                if debug_flag and consumed_now>0:
                    print(f"[DEBUG] drained={consumed_now} total_new={new_samples_since_train} replay_size={len(self.shared_replay) if self.shared_replay else 'n/a'}")
                # 追加デバッグ: 学習トリガ未達時に一定エピソードごとにラベル付サンプル比率を観測
                if debug_flag and (ep_done % max(1, int(self.config.get('debug_status_interval_eps', 25))) == 0):
                    try:
                        if self.shared_replay is not None:
                            total_rb = len(self.shared_replay)
                            labeled_rb = 0
                            try:
                                # shared_replay.iter_all で全要素にアクセスできる前提
                                for rec in self.shared_replay.iter_all():
                                    if isinstance(rec, dict) and rec.get('value') is not None:
                                        labeled_rb += 1
                            except Exception:
                                pass
                            print(f"[DEBUG][parent] ep_done={ep_done} shared_replay_total={total_rb} labeled={labeled_rb}")
                    except Exception:
                        pass

                # 新規サンプルがしきい値を超えたら学習を回す
                now = time.time()
                if new_samples_since_train >= min_new_samples_before_train:
                    # 学習バースト開始を可視化（長時間無音を避ける）
                    burst_start_ts = time.time()
                    last_loss_val = None
                    # updates_per_iter ステップだけ学習
                    for _ in range(max(1, int(updates_per_iter))):
                        loss_info = self.agents[0].train_step(batch_size=self.config.get("batch_size", 256))
                        try:
                            if isinstance(loss_info, dict) and loss_info.get("loss") is not None:
                                last_loss_val = float(loss_info.get("loss"))
                        except Exception:
                            pass
                        # 追加: モデル更新直後の即時保存+purge (超低メモリ運用オプション)
                        if self.config.get("purge_replay_after_each_update"):
                            try:
                                self._save_checkpoint()
                                # purge 後は学習トリガ用カウンタを 0 に (空なので)
                                new_samples_since_train = 0
                                if self.logger:
                                    self.logger.log_text("[replay] immediate_purge_after_update")
                            except Exception as _e:
                                print(f"[WARN] immediate save after update failed: {_e}")
                        # 学習回数カウント
                        train_it += 1
                        # 最小限の進捗表示
                        if self.minimal_progress and self.use_progress_bar:
                            if isinstance(loss_info, dict) and loss_info.get("loss") is not None:
                                loss_part = f"loss={loss_info['loss']:.4f}"
                            else:
                                loss_part = "loss=----"
                            lr_val = None
                            try:
                                opt = getattr(self.agents[0], '_optimizer', None)
                                if opt and hasattr(opt, 'param_groups') and opt.param_groups:
                                    lr_val = opt.param_groups[0].get('lr', None)
                            except Exception:
                                lr_val = None
                            lr_part = f"lr={lr_val:.2e}" if lr_val is not None else "lr=----"
                            line = f"[TRAIN~] it={train_it} {loss_part} {lr_part}"
                            print(line, end='\r', flush=True)
                        # ロガーへ
                        if self.logger and isinstance(loss_info, dict) and loss_info.get("loss") is not None and not getattr(self.agents[0], '_logged_inside', False):
                            self.logger.log_train(loss_info)
                        # 検証 (設定に応じて学習更新ごとに評価)
                        try:
                            val_every = int(self.config.get("val_eval_every_updates", 0) or 0)
                        except Exception:
                            val_every = 0
                        if self.logger and val_every > 0 and (train_it % val_every == 0):
                            try:
                                vinfo = self.agents[0].validate_step(batch_size=self.config.get("val_batch_size") or self.config.get("batch_size", 256))
                            except Exception:
                                vinfo = {"policy_loss": None, "value_loss": None, "entropy": None}
                            if isinstance(vinfo, dict) and (vinfo.get("policy_loss") is not None or vinfo.get("value_loss") is not None):
                                self.logger.log_validation(vinfo)
                        # ゲート途中経過ログは抑制（モデル更新時のみログ）
                    #  既存動作に合わせた保存は維持しつつ、高頻度I/Oを抑制
                    #  保存直前に評価ゲート（有効時）を実行し、採用可否でモデルを確定させる
                    def _resolve_device_str(dev_str: str | None) -> str:
                        if not dev_str or dev_str == "auto":
                            try:
                                import torch as _t
                                return "cuda" if _t.cuda.is_available() else "cpu"
                            except Exception:
                                return "cpu"
                        return dev_str

                    def _maybe_gate_before_save():
                        gate_enable = bool(self.config.get("eval_gate_enable", False))
                        if not gate_enable or self.model is None:
                            return None
                        # 起動ガード（学習更新回数ベース）
                        try:
                            _train_it = int(train_it)
                        except Exception:
                            _train_it = 0
                        try:
                            start_after = int(self.config.get("eval_gate_start_after_updates", 0) or 0)
                        except Exception:
                            start_after = 0
                        try:
                            every = int(self.config.get("eval_gate_every_updates", 0) or 0)
                        except Exception:
                            every = 0
                        if _train_it < start_after:
                            return None
                        if every > 0 and getattr(self, "_last_gate_start_it", -1) >= 0 and (_train_it - self._last_gate_start_it) < every:
                            return None
                        baseline_model = None
                        try:
                            from agents.models import PolicyValueNet as _PVN
                            # ベースラインは最新 ckpt を使用
                            base_path = self.config.get("checkpoint_path", "checkpoints/policy_value_latest.pt")
                            if os.path.exists(base_path):
                                dev_str = _resolve_device_str(self.config.get("device", None))
                                baseline_model = _PVN.load(base_path, map_location=dev_str)
                                try:
                                    dev = _resolve_device_str(self.config.get("device", None))
                                    if dev:
                                        baseline_model.to(dev)  # type: ignore[arg-type]
                                except Exception:
                                    pass
                        except Exception as _e:
                            if self.logger:
                                try:
                                    self.logger.log_text(f"[WARN] eval-gate baseline load failed (concurrent): {_e}")
                                except Exception:
                                    pass
                            baseline_model = None
                        if baseline_model is None:
                            return None
                        try:
                            from evaluation.gating import evaluate_candidate
                            games = int(self.config.get("eval_gate_games", 20) or 20)
                            thr = float(self.config.get("eval_gate_threshold", 0.6) or 0.6)
                            seed = self.config.get("eval_gate_seed", None)
                            # 評価ゲート開始ログ（同期）
                            try:
                                if self.logger:
                                    self.logger.log_text(f"[gate] start sync games={games} thr={thr:.0%} seed={seed}")
                                else:
                                    print(f"[gate] start sync games={games} thr={thr:.0%} seed={seed}")
                            except Exception:
                                pass
                            # 評価ゲート実行
                            # 起動時刻（更新番号）を記録
                            try:
                                self._last_gate_start_it = int(_train_it)
                            except Exception:
                                pass
                            gate_result = evaluate_candidate(self.model, baseline_model, self.config, games=games, seed=seed)
                            gated_pass = (gate_result.get("win_rate", 0.0) >= thr)
                            # 最新ゲート結果を保持（毎更新ログ用）
                            try:
                                self._last_gate_threshold = float(thr)
                                self._last_gate_result = dict(gate_result)
                            except Exception:
                                pass
                            # 結果ログ（合否と勝率を明示）
                            try:
                                wins = gate_result.get('wins')
                                total = gate_result.get('games')
                                decision = 'promote' if gated_pass else 'reject'
                                msg = f"[gate] result win_rate={gate_result.get('win_rate',0.0):.2%}"
                                if wins is not None and total is not None:
                                    msg += f" ({int(wins)}/{int(total)})"
                                msg += f" decision={decision} thr={thr:.0%}"
                                if self.logger:
                                    self.logger.log_text(msg)
                                else:
                                    print(msg)
                            except Exception:
                                pass
                            if not gated_pass:
                                # ロールバック
                                self.model = baseline_model
                                learner = self.agents[self.learning_player_id]
                                if isinstance(learner, AlphaZeroAgent):
                                    learner.set_model(self.model)
                            else:
                                # 合格時は以降の _save_checkpoint により latest が更新される
                                # ここでは昇格ログを出さない（結果ログに統合）
                                pass
                            return gate_result
                        except Exception as e:
                            if self.logger:
                                try:
                                    self.logger.log_text(f"[WARN] eval-gate failed in concurrent save: {e}")
                                except Exception:
                                    pass
                            return None

                    async_gate = bool(self.config.get("eval_gate_async", False))
                    if latest_ckpt_interval_sec > 0.0:
                        if (now - last_latest_ckpt_ts) >= latest_ckpt_interval_sec:
                            if async_gate:
                                # 学習更新回数に基づき起動をガード
                                if not self._should_start_gate(train_it):
                                    if self.logger:
                                        try:
                                            self.logger.log_text(f"[gate] skip: train_it={train_it}")
                                        except Exception:
                                            pass
                                else:
                                    # 候補モデルをステージングに保存し、ゲートを非同期起動（単発spawn）
                                    try:
                                        cand_path = os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "_candidate_eval.pt")
                                        # ベースラインは常に最新のチェックポイント（latest）
                                        base_path = self.config.get("checkpoint_path", "checkpoints/policy_value_latest.pt")
                                        if self.model is not None:
                                            try:
                                                self.model.save(cand_path, force_sync=True)
                                            except TypeError:
                                                self.model.save(cand_path)
                                        # 既存の評価が動いていないときのみ起動
                                        if getattr(self, "_gate_proc", None) is None or (self._gate_proc is not None and not self._gate_proc.is_alive()):
                                            games = int(self.config.get("eval_gate_games", 20) or 20)
                                            seed = self.config.get("eval_gate_seed", None)
                                            # 評価プロセスは常に CPU で実行
                                            dev_str = "cpu"
                                            self._gate_candidate_path = cand_path
                                            self._gate_baseline_path = base_path
                                            try:
                                                ctx = mp.get_context("spawn")
                                            except Exception:
                                                ctx = mp
                                            self._gate_queue = ctx.Queue()
                                            self._gate_proc = ctx.Process(target=_gate_worker_eval_proc, args=(cand_path, base_path, dict(self.config), games, seed, dev_str, self._gate_queue), daemon=True)
                                            self._gate_proc.start()
                                            # 起動記録
                                            try:
                                                self._last_gate_start_it = int(train_it)
                                            except Exception:
                                                pass
                                            # 非同期ゲート開始ログ
                                            try:
                                                thr = float(self.config.get("eval_gate_threshold", 0.6) or 0.6)
                                                if self.logger:
                                                    import os as _os
                                                    self.logger.log_text(
                                                        f"[gate] start async games={games} thr={thr:.0%} seed={seed} cand={_os.path.basename(cand_path)} base={_os.path.basename(base_path)}"
                                                    )
                                                else:
                                                    print(f"[gate] start async games={games} thr={thr:.0%} seed={seed} cand={cand_path} base={base_path}")
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                                # モデルはまだ昇格させず、リプレイ/メタのみ保存
                                self._save_checkpoint(skip_model_save=True)
                            else:
                                _ = _maybe_gate_before_save()
                                self._save_checkpoint()
                                if self.keep_prev_model:
                                    self._snapshot_current_model()
                            last_latest_ckpt_ts = now
                    else:
                        # 0 以下なら常に保存（従来挙動）
                        if async_gate:
                            # 学習更新回数に基づき起動をガード
                            if not self._should_start_gate(train_it):
                                if self.logger:
                                    try:
                                        self.logger.log_text(f"[gate] skip: train_it={train_it}")
                                    except Exception:
                                        pass
                            else:
                                try:
                                    cand_path = os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "_candidate_eval.pt")
                                    # ベースラインは常に最新のチェックポイント（latest）
                                    base_path = self.config.get("checkpoint_path", "checkpoints/policy_value_latest.pt")
                                    if self.model is not None:
                                        try:
                                            self.model.save(cand_path, force_sync=True)
                                        except TypeError:
                                            self.model.save(cand_path)
                                    # 既存の評価が動いていないときのみ起動
                                    if getattr(self, "_gate_proc", None) is None or (self._gate_proc is not None and not self._gate_proc.is_alive()):
                                        games = int(self.config.get("eval_gate_games", 20) or 20)
                                        seed = self.config.get("eval_gate_seed", None)
                                        # 評価プロセスは常に CPU で実行
                                        dev_str = "cpu"
                                        self._gate_candidate_path = cand_path
                                        self._gate_baseline_path = base_path
                                        try:
                                            ctx = mp.get_context("spawn")
                                        except Exception:
                                            ctx = mp
                                        self._gate_queue = ctx.Queue()
                                        self._gate_proc = ctx.Process(target=_gate_worker_eval_proc, args=(cand_path, base_path, dict(self.config), games, seed, dev_str, self._gate_queue), daemon=True)
                                        self._gate_proc.start()
                                        # 起動記録
                                        try:
                                            self._last_gate_start_it = int(train_it)
                                        except Exception:
                                            pass
                                        # 非同期ゲート開始ログ
                                        try:
                                            thr = float(self.config.get("eval_gate_threshold", 0.6) or 0.6)
                                            if self.logger:
                                                import os as _os
                                                self.logger.log_text(
                                                    f"[gate] start async games={games} thr={thr:.0%} seed={seed} cand={_os.path.basename(cand_path)} base={_os.path.basename(base_path)}"
                                                )
                                            else:
                                                print(f"[gate] start async games={games} thr={thr:.0%} seed={seed} cand={cand_path} base={base_path}")
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                            self._save_checkpoint(skip_model_save=True)
                        else:
                            _ = _maybe_gate_before_save()
                            self._save_checkpoint()
                            if self.keep_prev_model:
                                self._snapshot_current_model()
                    # ワーカー配布用モデルもスロットリングして保存
                    # 最新固定運用: ワーカー配布モデルは latest のみ。
                    # 学習中の候補モデルはワーカーへ配布しない（ゲート合格時のみ別経路で更新）。
                    # 学習トリガをリセット
                    new_samples_since_train = 0
                    # 学習バースト終了ログ
                    # 学習バースト終了の詳細ログは簡潔化のため削除

                # 定期ステータスログ
                now2 = time.time()
                if status_log_sec > 0 and (now2 - last_status_ts) >= status_log_sec:
                    last_status_ts = now2
                    # 非同期ゲートの完了をポーリング（あれば昇格/破棄を反映）
                    try:
                        self._poll_gate_async()
                    except Exception:
                        pass
                    try:
                        replay_size = len(self.shared_replay) if self.shared_replay is not None else (len(dst_buffer) if dst_buffer is not None and hasattr(dst_buffer, '__len__') else None)
                    except Exception:
                        replay_size = None
                    # 追加: 主要I/Oファイルサイズ (MB) を取得して status に添付
                    io_parts = []
                    try:
                        log_dir = self.config.get('log_dir', 'logs')
                        ev_path = os.path.join(log_dir, 'events.log')
                        mcts_path = os.path.join(log_dir, 'mcts_samples.jsonl')
                        ckpt_path = self.config.get('checkpoint_path', 'checkpoints/policy_value_latest.pt')
                        def _mb(p):
                            try:
                                return os.path.getsize(p) / (1024*1024)
                            except Exception:
                                return None
                        ev_mb = _mb(ev_path)
                        mcts_mb = _mb(mcts_path)
                        ckpt_mb = _mb(ckpt_path)
                        if ev_mb is not None:
                            io_parts.append(f"events:{ev_mb:.1f}MB")
                        if mcts_mb is not None:
                            io_parts.append(f"mcts:{mcts_mb:.1f}MB")
                        if ckpt_mb is not None:
                            io_parts.append(f"ckpt:{ckpt_mb:.1f}MB")
                    except Exception:
                        pass
                    io_sizes_str = (" io_sizes=" + ",".join(io_parts)) if io_parts else ""
                    # 追加: ワーカー生死と PID を簡単に添える
                    try:
                        procs_state = ",".join([f"{i}:{'A' if p.is_alive() else 'X'}@{p.pid}" for i,p in enumerate(procs)])
                    except Exception:
                        procs_state = ""
                    # 追記: ゲート評価プロセスの状態
                    try:
                        gp = getattr(self, "_gate_proc", None)
                        if gp is not None:
                            procs_state = (procs_state + ("," if procs_state else "")) + f"gate:{'A' if gp.is_alive() else 'X'}@{gp.pid}"
                    except Exception:
                        pass
                    msg = f"[status] ep_done={ep_done} train_it={train_it} new_since_train={new_samples_since_train} replay_size={replay_size} workers={workers}{io_sizes_str} procs=[{procs_state}]"
                    if self.logger:
                        self.logger.log_text(msg)
                        if status_log_include_mem:
                            try:
                                sample_cnt = replay_size
                                self.logger.log_memory_snapshot(sample_count=sample_cnt, force=False)
                            except Exception:
                                pass
                    else:
                        print(msg)

                # Shard の定期マージ (軽量: ここで低頻度に)
                if (time.time() - last_worker_rss_check) >= worker_rss_check_interval:
                    # RSS チェック前にマージを走らせる
                    try:
                        _merge_shards()
                    except Exception:
                        pass
                # Worker RSS チェック & 再起動
                if (time.time() - last_worker_rss_check) >= worker_rss_check_interval:
                    last_worker_rss_check = time.time()
                    try:
                        marked = _check_and_mark_restarts()
                        if marked:
                            _issue_graceful_restart(marked)
                    except Exception:
                        pass

            # 終了条件到達: ワーカー停止指示
            stop_event.set()
        finally:
            # 最後に少量の未学習サンプルが残っていれば 1 バーストだけ学習
            if new_samples_since_train > 0:
                for _ in range(max(1, int(updates_per_iter))):
                    _ = self.agents[0].train_step(batch_size=self.config.get("batch_size", 256))
                    # オプション: 最後の検証を1回実行（有効化時）
                    try:
                        val_every = int(self.config.get("val_eval_every_updates", 0) or 0)
                    except Exception:
                        val_every = 0
                    if self.logger and val_every > 0:
                        try:
                            vinfo = self.agents[0].validate_step(batch_size=self.config.get("val_batch_size") or self.config.get("batch_size", 256))
                        except Exception:
                            vinfo = {"policy_loss": None, "value_loss": None, "entropy": None}
                        if isinstance(vinfo, dict) and (vinfo.get("policy_loss") is not None or vinfo.get("value_loss") is not None):
                            self.logger.log_validation(vinfo)
                    if self.config.get("purge_replay_after_each_update"):
                        try:
                            self._save_checkpoint()
                            if self.logger:
                                self.logger.log_text("[replay] immediate_purge_after_update(final)")
                        except Exception as _e:
                            print(f"[WARN] immediate save after update (final) failed: {_e}")
            # 仕上げの保存（スロットリングに関わらず1回実施）
            # 最終保存前にも評価ゲート（有効時）を適用して確定モデルを選別
            try:
                gate_enable = bool(self.config.get("eval_gate_enable", False))
            except Exception:
                gate_enable = False
            if gate_enable and self.model is not None:
                baseline_model = None
                try:
                    from agents.models import PolicyValueNet as _PVN
                    # ベースラインは常に最新のチェックポイント（latest）
                    base_path = self.config.get("checkpoint_path", "checkpoints/policy_value_latest.pt")
                    if os.path.exists(base_path):
                        def _resolve_device_str(dev_str: str | None) -> str:
                            if not dev_str or dev_str == "auto":
                                try:
                                    import torch as _t
                                    return "cuda" if _t.cuda.is_available() else "cpu"
                                except Exception:
                                    return "cpu"
                            return dev_str
                        dev_str = _resolve_device_str(self.config.get("device", None))
                        baseline_model = _PVN.load(base_path, map_location=dev_str)
                        try:
                            dev = dev_str
                            if dev:
                                baseline_model.to(dev)  # type: ignore[arg-type]
                        except Exception:
                            pass
                except Exception as _e:
                    baseline_model = None
                if baseline_model is not None:
                    try:
                        from evaluation.gating import evaluate_candidate
                        games = int(self.config.get("eval_gate_games", 20) or 20)
                        thr = float(self.config.get("eval_gate_threshold", 0.6) or 0.6)
                        seed = self.config.get("eval_gate_seed", None)
                        # 最終ゲートは同期実行（終了直前のため）
                        gate_result = evaluate_candidate(self.model, baseline_model, self.config, games=games, seed=seed)
                        gated_pass = (gate_result.get("win_rate", 0.0) >= thr)
                        msg = f"[GATE] (final) win_rate={gate_result.get('win_rate'):.2%} threshold={thr:.0%} result={'ACCEPT' if gated_pass else 'REJECT'}"
                        # 最新ゲート結果を保持
                        try:
                            self._last_gate_threshold = float(thr)
                            self._last_gate_result = dict(gate_result)
                        except Exception:
                            pass
                        if self.logger:
                            self.logger.log_text(msg)
                        else:
                            print(msg)
                        if not gated_pass:
                            self.model = baseline_model
                            learner = self.agents[self.learning_player_id]
                            if isinstance(learner, AlphaZeroAgent):
                                learner.set_model(self.model)
                            if self.logger:
                                self.logger.log_text("[GATE] (final) reverted to baseline model")
                    except Exception:
                        pass
            self._save_checkpoint()
            if self.keep_prev_model and self.model is not None:
                self._snapshot_current_model()
            # 残りサンプルを吸い上げ
            _ = _drain_samples(max_items=None)
            # 最後にシャードをマージ (残骸取り込み)
            try:
                # 再利用: ローカル関数がスコープ外なら無視
                _merge_shards()
            except Exception:
                pass
            # ワーカー join
            for p in procs:
                try:
                    p.join(timeout=5)
                except Exception:
                    pass
            # 最終チェックポイント（念のため）
            self._save_checkpoint()
            # csv_summary_only の場合ここでサマリ行を書き出す
            try:
                if self.logger and getattr(self.logger, 'csv_summary_only', False):
                    self.logger.write_csv_summaries()
            except Exception as e:
                print(f"[WARN] write_csv_summaries(train_concurrent) failed: {e}")
        # 終了サマリを events.log に出力
        try:
            if self.logger:
                self.logger.log_text(f"[summary] total_episodes={ep_done} total_train_updates={train_it}")
                # バッファをフラッシュして確実にディスクへ書き込む
                if hasattr(self.logger, 'flush_buffers'):
                    self.logger.flush_buffers(force=True)
        except Exception:
            pass
        # サマリーを返す（合計エピソード数と総学習ステップ数）
        return {"episodes": ep_done, "train_updates": train_it}

    # ---------------- 並列自己対局 ----------------
    def _self_play_parallel(self, num_episodes: int, workers: int):
        """マルチプロセスで自己対局を並行実行し、共有リプレイへ集約する。

        注意:
          - Windows は spawn 方式のため、ワーカー関数はトップレベルに定義。
          - モデルは CPU コピーをミニチェックポイント経由で各ワーカーへ配布。
          - ワーカーはローカルバッファに確定サンプルのみを溜め、終了時に親へ返す。
        """
        start_time = time.time()
        # 進捗用
        total_done = 0
        last_progress_len = 0

        # モデル配布: 常に最新チェックポイントを使用
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        model_blob_path = self.config.get("checkpoint_path", os.path.join(self.config["checkpoint_dir"], "policy_value_latest.pt"))
        if (self.model is not None) and (not os.path.exists(model_blob_path)):
            try:
                self.model.save(model_blob_path)
            except Exception:
                pass

        remaining = int(num_episodes)
        ckpt_interval = int(self.ckpt_interval or 0)

        # 実行開始直後に 0% 進捗を表示（長いエピソードでも即表示）
        if self.minimal_progress and self.use_progress_bar and num_episodes > 0:
            bar, _pct = self._make_progress_bar(0, num_episodes)
            line = f"[SELFPLAY*] {bar} 0/{num_episodes} (workers={workers})"
            print(line, end='\r', flush=True)
            last_progress_len = len(line)

        while remaining > 0:
            # 次のチェックポイント境界までの残り (0=無効なら全量)
            if ckpt_interval > 0:
                to_next = ckpt_interval - (self._episodes_total_run % ckpt_interval)
                if to_next <= 0:
                    to_next = ckpt_interval
                chunk = min(remaining, to_next)
            else:
                chunk = remaining

            # このチャンクを workers にほぼ均等割当
            base = chunk // workers
            rem = chunk % workers
            ep_splits = [base + (1 if i < rem else 0) for i in range(workers)]
            tasks = []
            for wid, n_ep in enumerate(ep_splits):
                if n_ep <= 0:
                    continue
                tasks.append((wid, n_ep, self.config, model_blob_path))

            # 実行（spawn プールをチャンク毎に生成して安全に実行）
            results = []
            if tasks:
                with mp.get_context("spawn").Pool(processes=len(tasks)) as pool:
                    for wid, n_ep, cfg, model_path in tasks:
                        results.append(pool.apply_async(_selfplay_worker_entry, (wid, n_ep, cfg, model_path)))
                    # 逐次回収しリプレイへ反映
                    for res in results:
                        worker_out = res.get()
                        samples = worker_out.get("samples", [])
                        if self.shared_replay is None:
                            learner = self.agents[self.learning_player_id]
                            if isinstance(learner, AlphaZeroAgent):
                                for s in samples:
                                    learner.replay_buffer.append(s)
                        else:
                            for s in samples:
                                self.shared_replay.append(s)
                        total_done += worker_out.get("episodes", 0)
                        if self.minimal_progress and self.use_progress_bar:
                            bar, _pct = self._make_progress_bar(total_done, num_episodes)
                            line = f"[SELFPLAY*] {bar} {total_done}/{num_episodes} (workers={workers})"
                            pad = max(0, last_progress_len - len(line))
                            print(line + ' ' * pad, end='\r' if total_done < num_episodes else '\n', flush=True)
                            last_progress_len = len(line)

            # エピソードカウンタを 1 ずつ進めて単一実行時と同じタイミングのフックを発火
            for _ in range(chunk):
                self._episodes_total_run += 1
                # 周期チェックポイント保存（単一実行時と同一判定）
                if self.ckpt_interval > 0 and (self._episodes_total_run % self.ckpt_interval == 0):
                    self._save_checkpoint(version_tag=f"ep{self._episodes_total_run}")
                    if self.keep_prev_model:
                        self._snapshot_current_model()
                    if self.keep_prev_model and self.prev_model_mix_players > 0:
                        if self.past_models:
                            self._assign_past_models_to_opponents()
                        elif self._previous_model is not None:
                            self._mix_previous_model_opponents()
                    # モデル世代を進め、エージェントへ反映
                    self.model_version += 1
                    for ag in self.agents:
                        if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'model_version'):
                            ag.model_version = self.model_version
                # 任意のエピソード間隔で opponent 再割当
                if self.opponent_mix_interval > 0 and self.keep_prev_model and self.prev_model_mix_players > 0:
                    if (self._episodes_total_run % self.opponent_mix_interval == 0) and self.past_models:
                        self._assign_past_models_to_opponents()

            remaining -= chunk

        # 終了ログ
        elapsed = time.time() - start_time
        if not self.minimal_progress:
            print(f"[SELFPLAY*] finished {num_episodes} episodes in {elapsed:.1f}s using {workers} workers")
        return

    # -----------------------------------------------------
    # 過去モデルスナップショット
    # -----------------------------------------------------
    def _snapshot_current_model(self):
        """現在モデルを CPU 上に複製し past_models に追加。容量上限を超えたら FIFO で削除。

        注意: self._previous_model (旧互換) も最新スナップショットとして更新し、
              プール未使用設定時のフォールバックを維持する。
        """
        if self.model is None:
            return
        try:
            # live モデル state_dict を CPU 上にコピー
            # フル特徴量モデルの場合は input_dim を揃える必要がある (そうでないと 246->6 などの shape mismatch が発生)
            use_full = getattr(self.model, 'use_full_features', False)
            # v4 特徴: 59N + 125。旧式 (56N+22) に固定していたため mismatch が発生していたので、
            # 現行モデルの in_features をそのまま利用してスナップショットを生成する。
            full_dim = None
            if use_full:
                try:
                    full_dim = int(getattr(self.model.backbone[0], 'in_features'))  # type: ignore[index]
                except Exception:
                    full_dim = None
            if use_full and full_dim is None:
                # それでも取得不能な場合は v4 期待式で推定 (警告付き)
                try:
                    est = 59 * int(self.config.get("num_players", 4)) + 125
                    print(f"[snapshot][WARN] full_dim 推定 fallback -> {est}")
                    full_dim = est
                except Exception:
                    pass
            try:
                snap = PolicyValueNet(
                    max_policy_size=self.config["max_policy_size"],
                    hidden_size=self.config["hidden_size"],
                    num_players=self.config["num_players"],
                    device="cpu",
                    use_full_features=use_full,
                    full_feature_dim=full_dim if use_full else None,
                )
                snap.load_state_dict(self.model.state_dict(), strict=True)  # type: ignore[arg-type]
            except Exception as e_build:
                # 最終フォールバック: 既存モデルを deepcopy して CPU へ移動（互換重視 / shape mismatch 無視）
                import copy as _cp
                print(f"[snapshot][WARN] 標準スナップショット再構築失敗 -> deepcopy fallback ({type(e_build).__name__}: {e_build})")
                snap = _cp.deepcopy(self.model)
                try:
                    snap.to('cpu')
                except Exception:
                    pass
            self.past_models.append(snap)
            self._previous_model = snap  # 互換
            # 上限超過なら古いものから削除
            if self.past_model_pool_size > 0 and len(self.past_models) > self.past_model_pool_size:
                overflow = len(self.past_models) - self.past_model_pool_size
                if overflow > 0:
                    del self.past_models[0:overflow]
            if self.logger:
                self.logger.log_text(f"[snapshot] pool_size={len(self.past_models)}")
        except Exception as e:
            if self.logger:
                self.logger.log_text(f"[WARN] snapshot failed: {e}")
            else:
                print(f"[WARN] snapshot failed: {e}")

    # -----------------------------------------------------
    # 過去モデルを対戦相手へランダム割当
    # -----------------------------------------------------
    def _assign_past_models_to_opponents(self):
        if not self.past_models:
            return
        candidate_indices = [i for i in range(self.config["num_players"]) if i != self.learning_player_id]
        if not candidate_indices:
            return
        random.shuffle(candidate_indices)
        selected = candidate_indices[: self.prev_model_mix_players]
        # 割当: 選択プレイヤーにランダムな過去モデルを割り当てる
        for idx in selected:
            ag = self.agents[idx]
            if isinstance(ag, AlphaZeroAgent):
                model_choice = random.choice(self.past_models)
                ag.set_model(model_choice)
        # 学習プレイヤーは常に最新モデル
        learner = self.agents[self.learning_player_id]
        if isinstance(learner, AlphaZeroAgent):
            learner.set_model(self.model)
        if self.logger:
            self.logger.log_text(f"[mix-multi] assigned past models to players={selected} (pool={len(self.past_models)})")
        else:
            print(f"[INFO] past models mixed into players={selected} (pool={len(self.past_models)})")
        if self.minimal_progress:
            # ループ終了で改行確定
            if not self.use_progress_bar:
                print()

    def _play_one_episode(self, episode_index: int):
        # 環境を初期化 (DaifugoSimpleEnv が reset を持つ想定)
        if hasattr(self.env, "reset"):
            self.env.reset()
        # エピソード内フェーズ勝利カウント (学習プレイヤー視点)
        phase_wins = 0
        phase_attempts = 0
        # 各学習エージェントの move カウンタをリセット
        for ag in self.agents:
            if isinstance(ag, AlphaZeroAgent) and hasattr(ag, 'reset_episode'):
                ag.reset_episode()
        # ゲーム進行ループ (env.game.done を監視)
        step_count = 0
        prev_rankings: List[int] = list(getattr(self.env.game, "rankings", []))
        max_steps = self.config.get("max_episode_steps", 1000)
        while not getattr(self.env.game, "done", False):
            if step_count >= max_steps:
                print(f"[WARN] episode_index={episode_index} step_limit_reached={max_steps} -> force_terminate")
                break
            current_player_id = self.env.game.turn
            agent = self.agents[current_player_id]
            # 学習エージェントと簡易エージェントで呼び出し方法を分岐
            if isinstance(agent, AlphaZeroAgent):
                action = agent.select_action(self.env, training=True)
            else:
                # 観測と合法手を取得しシンプルエージェントへ渡す
                current_player = self.env.game.players[self.env.game.turn]
                hand = current_player.hand
                field = self.env.game.current_field[:]
                legal_actions = self.env._generate_legal_actions(hand, field)
                obs_simple = {'hand': hand, 'field': field}
                action = agent.select_action(obs_simple, legal_actions=legal_actions)
            # 行動適用
            # パス表現は None に統一 (環境側 step で None を直接パス処理できるようになった)
            ext_act = action  # action が None ならそのままパス
            try:
                self.env.step(external_action=ext_act)
            except TypeError:
                self.env.step(ext_act)
            step_count += 1
            # フェーズ(誰かが新たに上がった)検知
            current_rankings: List[int] = list(getattr(self.env.game, "rankings", []))
            if len(current_rankings) > len(prev_rankings):
                # 新規に上がったプレイヤー(複数同時も許容)
                new_winners = current_rankings[len(prev_rankings):]
                for winner_id in new_winners:
                    for ag in self.agents:
                        if isinstance(ag, AlphaZeroAgent):
                            was_active = ag.player_id not in prev_rankings  # 以前まだ上がっていなかったか
                            # 学習プレイヤー視点のフェーズ統計更新
                            if ag.player_id == self.learning_player_id and was_active:
                                phase_attempts += 1
                                if winner_id == self.learning_player_id:
                                    phase_wins += 1
                            ag.finalize_phase(winner_player_id=winner_id, was_active=was_active)
                prev_rankings = current_rankings
        # 念のため未確定フェーズを 0 でクリア
        for ag in self.agents:
            if isinstance(ag, AlphaZeroAgent):
                ag.flush_unfinished_phase()
    # ステップ上限で打ち切られた場合も flush 済なのでそのまま終了
        # 最終順位ベース報酬は付与しない方針 (finalize_game は no-op)
        for ag in self.agents:
            if isinstance(ag, AlphaZeroAgent):
                ag.finalize_game()
        # Episode メトリクス集計
        rankings = list(getattr(self.env.game, 'rankings', []))
        episode_len = step_count
        avg_rank = None
        first_rate = 0.0
        if rankings:
            # 学習プレイヤー順位 (1-based)
            if self.learning_player_id in rankings:
                avg_rank = rankings.index(self.learning_player_id) + 1
            first_rate = 1.0 if rankings and rankings[0] == self.learning_player_id else 0.0
        # 学習プレイヤーの累積フェーズ勝率 (Agent内のカウンタ) 取得
        learner_agent = self.agents[self.learning_player_id]
        cum_phase_rate = None
        if isinstance(learner_agent, AlphaZeroAgent) and learner_agent.total_value_samples > 0:
            cum_phase_rate = learner_agent.total_positive / learner_agent.total_value_samples

        phase_win_rate = (phase_wins / phase_attempts) if phase_attempts > 0 else None
        # フェーズ予測精度 (学習プレイヤーでのみ定義)
        phase_acc = None
        if isinstance(learner_agent, AlphaZeroAgent) and learner_agent.episode_phase_total > 0:
            phase_acc = learner_agent.episode_phase_correct / learner_agent.episode_phase_total
        ep_metrics = {
            "avg_rank": avg_rank,
            "first_rate": first_rate,
            "episode_len": episode_len,
            "phase_acc": phase_acc,
            "phase_win_rate": phase_win_rate,
            "phase_wins": phase_wins,
            "phase_attempts": phase_attempts,
            "cum_phase_win_rate": cum_phase_rate,
        }
        if self.logger:
            self.logger.log_episode(ep_metrics)

    # 旧最終順位一括報酬方式は廃止 (フェーズごとの finalize_phase を利用)
    def _finalize_episode_rewards(self):  # 互換: 何もしない
        return

    # -----------------------------------------------------
    # モデル学習 (ダミー)
    # -----------------------------------------------------
    def train_updates(self, num_updates: int = 1):
        start_time = time.time()
        last_print = 0
        successful_updates = 0  # 実際に損失が計算できた学習回数
        # --- 評価ゲート用: 学習前のベースラインモデルを保持（存在すれば ckpt、なければ現モデルのコピー） ---
        gate_enable = bool(self.config.get("eval_gate_enable", False))
        baseline_model = None
        if gate_enable:
            try:
                from agents.models import PolicyValueNet as _PVN
                # ベースラインは最新 ckpt
                base_ckpt = self.config.get("checkpoint_path", "checkpoints/policy_value_latest.pt")
                if os.path.exists(base_ckpt):
                    # resolve device 'auto' -> actual
                    def _resolve_device_str(dev_str: str | None) -> str:
                        if not dev_str or dev_str == "auto":
                            try:
                                import torch as _t
                                return "cuda" if _t.cuda.is_available() else "cpu"
                            except Exception:
                                return "cpu"
                        return dev_str
                    dev_str = _resolve_device_str(self.config.get("device", None))
                    baseline_model = _PVN.load(base_ckpt, map_location=dev_str)
                    # align device if needed
                    try:
                        if dev_str:
                            baseline_model.to(dev_str)  # type: ignore[arg-type]
                    except Exception:
                        pass
                else:
                    # フォールバック: 現在モデルの shallow コピー（同一参照回避）
                    import copy as _copy
                    baseline_model = _copy.deepcopy(self.model)
            except Exception as _e:
                print(f"[WARN] eval-gate baseline load failed: {_e}")
                baseline_model = None
        for i in range(num_updates):
            loss_info = self.agents[0].train_step(batch_size=self.config.get("batch_size", 256))
            # ログ用にエポック情報を付与（存在する辞書に無害に追加）
            if isinstance(loss_info, dict):
                loss_info.setdefault("total_epochs", num_updates)
            # サンプル不足/偏り警告
            if loss_info.get("reason") == "no_data":
                print("[WARN] train_step skipped: no_data (consider increasing episodes or buffer)")
            else:
                if isinstance(loss_info, dict) and loss_info.get("loss") is not None:
                    successful_updates += 1
                # 共有バッファ存在時に学習プレイヤーサンプルの割合を軽くチェック
                if self.config.get("use_shared_replay", False) and self.shared_replay is not None:
                    total = len(self.shared_replay)
                    if total > 0:
                        own = sum(1 for _ in self.shared_replay.iter_all(owner_pid=self.learning_player_id))
                        ratio = own / total
                        if ratio < 0.15:  # しきい値は暫定
                            print(f"[WARN] low data share for learner: {own}/{total} ({ratio:.2%})")
            # 進捗出力 (動的バー)
            if self.minimal_progress and self.use_progress_bar:
                done = i + 1
                # シンプル表示: バー無し / ETA無し / 一行上書き
                if isinstance(loss_info, dict) and loss_info.get("loss") is not None:
                    loss_part = f"loss={loss_info['loss']:.4f}"
                else:
                    loss_part = "loss=----"
                lr_val = None
                try:
                    opt = getattr(self.agents[0], '_optimizer', None)
                    if opt and hasattr(opt, 'param_groups') and opt.param_groups:
                        lr_val = opt.param_groups[0].get('lr', None)
                except Exception:
                    lr_val = None
                lr_part = f"lr={lr_val:.2e}" if lr_val is not None else "lr=----"
                line = f"[TRAIN] {done}/{num_updates} {loss_part} {lr_part}"
                pad = max(0, self._last_progress_len - len(line))
                print(line + ' ' * pad, end='\r' if done < num_updates else '\n', flush=True)
                self._last_progress_len = len(line)
            else:
                if (i + 1) % self.config.get("log_interval", 50) == 0:
                    # 標準出力に加えて events.log にも同一行を追記
                    _msg = f"[TRAIN] epoch={i+1}/{num_updates} loss={loss_info}"
                    try:
                        if self.logger is not None:
                            # 既に print も行うため also_print は False
                            self.logger.log_text(_msg, also_print=False)
                    except Exception:
                        pass
                    print(_msg)
            # ロガーへ (train_step 内で既に push されている場合は二重記録を避ける)
            if self.logger and loss_info.get("loss") is not None and not getattr(self.agents[0], '_logged_inside', False):
                # 新列: 非並行モードではローカルカウンタで代用（logger 側で update_step をフォールバックに使用）
                try:
                    payload = dict(loss_info)
                    payload["train_count"] = i + 1
                except Exception:
                    payload = loss_info
                self.logger.log_train(payload)

            # 検証: 指定間隔で検証バッチの損失を測定しログへ
            try:
                val_every = int(self.config.get("val_eval_every_updates", 0) or 0)
            except Exception:
                val_every = 0
            if self.logger and val_every > 0 and ((i + 1) % val_every == 0):
                try:
                    vinfo = self.agents[0].validate_step(batch_size=self.config.get("val_batch_size") or self.config.get("batch_size", 256))
                except Exception:
                    vinfo = {"policy_loss": None, "value_loss": None, "entropy": None}
                # 可能ならロギング
                if isinstance(vinfo, dict) and (vinfo.get("policy_loss") is not None or vinfo.get("value_loss") is not None):
                    self.logger.log_validation(vinfo)
            # 追加: 各学習更新後に最新のゲート結果を毎回ログ（非並行モード用）
            try:
                thr_val = float(self.config.get("eval_gate_threshold", self._last_gate_threshold)) if hasattr(self, "_last_gate_threshold") else 0.6
            except Exception:
                thr_val = 0.6
            try:
                last_res = getattr(self, "_last_gate_result", None)
                if self.logger and isinstance(last_res, dict) and (last_res.get("win_rate") is not None):
                    wr = float(last_res.get("win_rate", 0.0))
                    msg_gate = f"[GATE] win_rate={wr:.2%} threshold={thr_val:.0%} result={'ACCEPT' if wr >= thr_val else 'REJECT'}"
                    self.logger.log_text(msg_gate)
            except Exception:
                pass

            # 即時 checkpoint + purge オプション (単独 train_updates 用 / concurrent とは別経路)
            if self.config.get('purge_replay_after_each_update'):
                try:
                    # purge は _save_checkpoint 内の設定に依存 (purge_replay_after_checkpoint)
                    # 事前にサイズ計測
                    old_size = None
                    try:
                        if self.shared_replay is not None:
                            old_size = len(self.shared_replay)
                        else:
                            ag0 = self.agents[0]
                            rb = getattr(ag0, 'replay_buffer', None)
                            if rb is not None and hasattr(rb, '__len__'):
                                old_size = len(rb)
                    except Exception:
                        old_size = None
                    self._save_checkpoint()
                    if self.logger:
                        self.logger.log_text(f"[replay] immediate_purge_after_update(train_updates) prev_size={old_size}")
                except Exception as e:
                    print(f"[WARN] immediate checkpoint after update failed: {e}")
        # --- 学習後: ゲート評価（有効時） ---
        gated_pass = True
        gate_result = None
        if gate_enable and (self.model is not None) and (baseline_model is not None):
            try:
                from evaluation.gating import evaluate_candidate
                games = int(self.config.get("eval_gate_games", 20) or 20)
                thr = float(self.config.get("eval_gate_threshold", 0.6) or 0.6)
                seed = self.config.get("eval_gate_seed", None)
                gate_result = evaluate_candidate(self.model, baseline_model, self.config, games=games, seed=seed)
                gated_pass = (gate_result.get("win_rate", 0.0) >= thr)
                msg = f"[GATE] win_rate={gate_result.get('win_rate'):.2%} threshold={thr:.0%} result={'ACCEPT' if gated_pass else 'REJECT'}"
                # 最新ゲート結果を保持
                try:
                    self._last_gate_threshold = float(thr)
                    self._last_gate_result = dict(gate_result)
                except Exception:
                    pass
                if self.logger:
                    self.logger.log_text(msg)
                else:
                    print(msg)
            except Exception as e:
                print(f"[WARN] eval-gate failed to run: {e}")
                gated_pass = True  # フォールバックで通す

        # 許可された場合のみ保存。拒否ならロールバックして旧モデルを維持
        if not gated_pass and baseline_model is not None:
            try:
                # ロールバック
                self.model = baseline_model
                # 学習プレイヤーへも反映
                learner = self.agents[self.learning_player_id]
                if isinstance(learner, AlphaZeroAgent):
                    learner.set_model(self.model)
                if self.logger:
                    self.logger.log_text("[GATE] reverted to baseline model")
            except Exception as e:
                print(f"[WARN] revert to baseline failed: {e}")

        # 保存実行（gated_pass に応じて self.model は適切な方がセットされている）
        self._save_checkpoint()
        # 学習直後の最新モデルもスナップショット (自己対局前に世代差が明確になる)
        if self.keep_prev_model:
            self._snapshot_current_model()
        # CSV サマリーモードならここで 1 行のみ書き出し
        if self.logger and getattr(self.logger, 'csv_summary_only', False):
            try:
                self.logger.write_csv_summaries()
            except Exception as e:
                print(f"[WARN] write_csv_summaries failed: {e}")
        # 終了サマリを events.log に出力
        try:
            if self.logger:
                extra = ''
                try:
                    if gate_result is not None:
                        extra = f" gate_win_rate={gate_result.get('win_rate'):.4f} gate_games={gate_result.get('games')}"
                except Exception:
                    extra = ''
                self.logger.log_text(f"[summary] train_updates_requested={num_updates} train_updates_successful={successful_updates}{extra}")
                # バッファをフラッシュして確実にディスクへ書き込む
                if hasattr(self.logger, 'flush_buffers'):
                    self.logger.flush_buffers(force=True)
        except Exception:
            pass

    # -----------------------------------------------------
    # 進捗バー生成ヘルパー
    # -----------------------------------------------------
    def _make_progress_bar(self, done: int, total: int, pct: float | None = None):
        width = max(10, int(self.progress_bar_width))
        ratio = 0.0 if total <= 0 else min(1.0, max(0.0, done / total))
        fill = int(ratio * width)
        bar = "#" * fill + "-" * (width - fill)
        pct_val = ratio * 100.0
        return f"[{bar}] {pct_val:6.2f}%", f"{pct_val:6.2f}%"

    # -----------------------------------------------------
    # チェックポイント
    # -----------------------------------------------------
    def _save_checkpoint(self):  # backward compatibility name
        self._save_checkpoint(version_tag=None)

    def _save_checkpoint(self, version_tag: str | None = None, *, model_path_override: str | None = None, skip_model_save: bool = False):
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        # アトミック保存ヘルパ
        def _atomic_save(fn, save_callable):
            tmp = fn + ".tmp"
            try:
                save_callable(tmp)
                os.replace(tmp, fn)
            except Exception as e:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                print(f"[WARN] atomic save fallback ({fn}): {e}")
                try:
                    save_callable(fn)
                except Exception as ee:
                    print(f"[ERROR] save failed {fn}: {ee}")

        # 最新モデル保存 (atomic)
        latest_path = self.config["checkpoint_path"] if model_path_override is None else model_path_override
        if (not skip_model_save) and self.model is not None:
            _atomic_save(latest_path, lambda p: self.model.save(p, force_sync=True))
        # バージョン付き保存
        if version_tag:
            ver_path = os.path.join(self.config["checkpoint_dir"], f"policy_value_{version_tag}.pt")
            if (not skip_model_save) and self.model is not None:
                _atomic_save(ver_path, lambda p: self.model.save(p, force_sync=True))
        # リプレイ保存: 共有モードなら共有バッファを保存、そうでなければ従来通り代表エージェント
        replay_path = self.config.get("replay_path", "replay_buffer.joblib")
        purge_after_ckpt = bool(self.config.get("purge_replay_after_checkpoint", False))
        if self.shared_replay is not None:
            try:
                # フル特徴量モードで legacy 混入していたらフィルタ
                if self.config.get('use_full_features'):
                    try:
                        before = len(self.shared_replay)
                        # 共有リプレイ内部構造アクセス (互換維持: data 属性が存在しない将来版対策)
                        data_attr = getattr(self.shared_replay, '_data', None)
                        if data_attr is not None:
                            # _data が deque の場合は直接フィルタせず (コスト高)、ここではスキップ
                            pass
                        else:
                            # 旧バージョン data フィールド互換 (安全性低いため try 内で)
                            self.shared_replay.data = [s for s in self.shared_replay.data if s.get('feature_version',1)==1]  # type: ignore[attr-defined]
                        after = len(self.shared_replay)
                        if after < before:
                            print(f"[INFO] shared replay purge legacy {before-after} samples (full mode)")
                    except Exception:
                        pass
                self.shared_replay.save(replay_path, purge=purge_after_ckpt)
                if purge_after_ckpt:
                    if self.logger:
                        try:
                            self.logger.log_text(f"[replay] checkpoint_saved_and_purged prev_size={before} new_size=0 path={replay_path}")
                        except Exception:
                            pass
            except Exception as e:
                print(f"[WARN] shared replay save failed: {e}")
        else:
            try:
                # 単独エージェント経由保存
                ag0 = self.agents[0]
                if hasattr(ag0, 'save_replay'):
                    prev_size = None
                    try:
                        rb_local = getattr(ag0, 'replay_buffer', None)
                        if rb_local is not None and hasattr(rb_local, '__len__'):
                            prev_size = len(rb_local)
                    except Exception:
                        prev_size = None
                    ag0.config['purge_replay_after_save'] = purge_after_ckpt
                    ag0.save_replay(replay_path)
                    if purge_after_ckpt and self.logger:
                        self.logger.log_text(f"[replay] agent_replay_saved_and_purged prev_size={prev_size} path={replay_path}")
            except Exception as e:
                print(f"[WARN] agent replay save failed: {e}")
        # Optimizer 保存（学習エージェントの Adam 状態）
        try:
            ag0 = self.agents[self.learning_player_id]
            opt_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'optimizer_latest.pt')
            def _save_opt(pth):
                if hasattr(ag0, 'save_optimizer'):
                    ag0.save_optimizer(pth)
                else:
                    import torch as _t
                    opt = getattr(ag0, '_optimizer', None)
                    if opt is not None:
                        _t.save(opt.state_dict(), pth)
            _atomic_save(opt_path, _save_opt)
        except Exception as e:
            print(f"[WARN] optimizer save failed: {e}")
        # Scheduler 保存（学習率スケジューラの状態）
        try:
            ag0 = self.agents[self.learning_player_id]
            # スケジューラが存在する場合のみ保存
            if hasattr(ag0, '_scheduler') and getattr(ag0, '_scheduler') is not None:
                sch_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'scheduler_latest.pt')
                def _save_sch(pth):
                    if hasattr(ag0, 'save_scheduler'):
                        ag0.save_scheduler(pth)
                _atomic_save(sch_path, _save_sch)
        except Exception as e:
            print(f"[WARN] scheduler save failed: {e}")
        # メタデータ保存
        try:
            # 簡易 config ハッシュ
            import json, hashlib, time as _time
            cfg_sorted = json.dumps(self.config, sort_keys=True, ensure_ascii=False).encode('utf-8')
            cfg_hash = hashlib.md5(cfg_sorted).hexdigest()
            meta = {
                "model_version": self.model_version,
                "use_full_features": bool(self.config.get('use_full_features')),\
                "num_players": self.config.get('num_players'),
                "hidden_size": self.config.get('hidden_size'),
                "max_policy_size": self.config.get('max_policy_size'),
                "seed": self.config.get('seed'),
                "config_md5": cfg_hash,
            }
            try:
                if self.model is not None and hasattr(self.model, 'backbone'):
                    meta['full_feature_dim'] = getattr(self.model.backbone[0], 'in_features', None)
            except Exception:
                pass
            meta['saved_at'] = _time.time()
            meta_path = os.path.join(self.config['checkpoint_dir'], 'metadata.json')
            def _write_meta(p):
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            # atomic 書き込み
            tmp_meta = meta_path + '.tmp'
            try:
                _write_meta(tmp_meta)
                os.replace(tmp_meta, meta_path)
            except Exception:
                try:
                    if os.path.exists(tmp_meta):
                        os.remove(tmp_meta)
                except Exception:
                    pass
                _write_meta(meta_path)
        except Exception as e:
            print(f"[WARN] metadata save failed: {e}")

    # 即時保存用ヘルパ (ユーザーが強制的に現在状態を吐き出したい場合)
    def force_save(self):
        try:
            self._save_checkpoint()
            if self.logger:
                self.logger.log_text("[force_save] checkpoint+replay saved")
        except Exception as e:
            print(f"[WARN] force_save failed: {e}")

    # -----------------------------------------------------
    # 非同期ゲート: ポーリングして結果を反映（合格で昇格/不合格で破棄）
    # -----------------------------------------------------
    def _poll_gate_async(self):
        # 単発spawn方式: プロセスとキューから非ブロッキングで回収
        proc = getattr(self, "_gate_proc", None)
        q = getattr(self, "_gate_queue", None)
        if proc is None:
            return
        # まだ動作中なら結果は未到着の可能性あり
        if proc.is_alive():
            # キューに結果が既にある場合もあるので試す（非ブロッキング）
            try:
                res = q.get_nowait() if q is not None else None
            except _queue.Empty:
                return
            except Exception:
                res = None
        else:
            # プロセスが停止しているなら結果取得を試みる
            try:
                res = q.get_nowait() if q is not None else None
            except Exception:
                res = None
        # プロセスのクリーンアップ
        try:
            if not proc.is_alive():
                proc.join(timeout=1)
        except Exception:
            pass
        self._gate_proc = None
        self._gate_queue = None
        if res is None:
            # 失敗として扱う（ログのみ）
            if self.logger:
                try:
                    self.logger.log_text("[WARN] eval-gate async finished without result")
                except Exception:
                    pass
            return
        if isinstance(res, dict) and "error" in res:
            if self.logger:
                try:
                    self.logger.log_text(f"[WARN] eval-gate async failed: {res['error']}")
                except Exception:
                    pass
            return
        cand = self._gate_candidate_path
        base = self._gate_baseline_path
        self._gate_candidate_path = None
        self._gate_baseline_path = None
        try:
            thr = float(self.config.get("eval_gate_threshold", 0.6) or 0.6)
        except Exception:
            thr = 0.6
        win_rate = float(res.get("win_rate", 0.0)) if isinstance(res, dict) else 0.0
        gated_pass = (win_rate >= thr)
        # 最新ゲート結果を保持（評価途中の詳細ログは抑制）
        try:
            self._last_gate_threshold = float(thr)
            # res は dict 想定
            if isinstance(res, dict):
                self._last_gate_result = dict(res)
            else:
                self._last_gate_result = {"win_rate": float(win_rate)}
        except Exception:
            pass
        # 結果ログ（非同期完了時）
        try:
            decision = 'promote' if gated_pass else 'reject'
            wins = None
            total = None
            try:
                wins = int(res.get('wins')) if isinstance(res, dict) and res.get('wins') is not None else None
                total = int(res.get('games')) if isinstance(res, dict) and res.get('games') is not None else None
            except Exception:
                wins, total = None, None
            import os as _os
            cand_name = _os.path.basename(cand) if cand else None
            base_name = _os.path.basename(base) if base else None
            msg = f"[gate] done async win_rate={win_rate:.2%}"
            if wins is not None and total is not None:
                msg += f" ({wins}/{total})"
            msg += f" decision={decision} thr={thr:.0%}"
            if cand_name and base_name:
                msg += f" cand={cand_name} base={base_name}"
            if self.logger:
                self.logger.log_text(msg)
            else:
                print(msg)
        except Exception:
            pass
        # 昇格 or 破棄
        try:
            if gated_pass and cand is not None:
                latest_path = self.config.get("checkpoint_path", "checkpoints/policy_value_latest.pt")
                # 候補を本番(最新)へ昇格
                os.replace(cand, latest_path)
                # スナップショット等の付随処理
                if self.keep_prev_model:
                    try:
                        self._snapshot_current_model()
                    except Exception:
                        pass
            else:
                # 拒否時は候補を削除
                if cand is not None and os.path.exists(cand):
                    try:
                        os.remove(cand)
                    except Exception:
                        pass
        except Exception:
            pass

    # -----------------------------------------------------
    # 直前モデルを対戦相手に混在させる
    # -----------------------------------------------------
    def _mix_previous_model_opponents(self):
        if self._previous_model is None:
            return
        # 学習プレイヤー以外から N 人を選び、そのエージェントの model を previous_model に差し替える
        candidate_indices = [i for i in range(self.config["num_players"]) if i != self.learning_player_id]
        if not candidate_indices:
            return
        random.shuffle(candidate_indices)
        selected = candidate_indices[: self.prev_model_mix_players]
        for idx in selected:
            ag = self.agents[idx]
            if isinstance(ag, AlphaZeroAgent):
                ag.set_model(self._previous_model)
        # 学習プレイヤーのモデルは必ず最新 (self.model) に戻しておく
        learner = self.agents[self.learning_player_id]
        if isinstance(learner, AlphaZeroAgent):
            learner.set_model(self.model)
        if self.logger:
            self.logger.log_text(f"[mix] applied previous model to players={selected}")
        else:
            print(f"[INFO] previous model mixed into players={selected}")


__all__ = ["Trainer"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero 大富豪 小規模学習")
    parser.add_argument("--episodes", type=int, default=3, help="自己対局エピソード数 (default: 3)")
    parser.add_argument("--updates", type=int, default=5, help="train_step 呼び出し回数 (default: 5)")
    parser.add_argument("--num-sim", type=int, default=None, help="MCTS シミュレーション数を一時的に上書き")
    parser.add_argument("--batch-size", type=int, default=None, help="train_step のバッチサイズ一時上書き")
    parser.add_argument("--log-dir", type=str, default=None, help="ログディレクトリ上書き")
    parser.add_argument("--no-tb", action="store_true", help="TensorBoard を無効化")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード上書き")
    parser.add_argument("--device", type=str, default=None, help="使用デバイスを指定 (cpu/cuda/cuda:0)。未指定はauto")
    parser.add_argument("--workers", type=int, default=None, help="自己対局の並列ワーカー数。未指定は設定値を使用")
    # 並行実行オプション
    parser.add_argument("--concurrent", action="store_true", help="自己対局と学習を並行実行する常駐モードを有効化")
    parser.add_argument("--concurrent-updates-per-iter", type=int, default=50, help="並行モードでの学習ステップ束ね数")
    parser.add_argument("--concurrent-queue-size", type=int, default=50000, help="並行モードでのサンプルQueueの最大長")
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=None,
        help="並行モード(--concurrent)での自己対局の総エピソード上限",
    )
    args = parser.parse_args()

    cfg = dict(ALPHA_ZERO_CONFIG)
    if args.num_sim is not None:
        cfg["num_simulations"] = args.num_sim
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.log_dir is not None:
        cfg["log_dir"] = args.log_dir
    if args.no_tb:
        cfg["enable_tensorboard"] = False
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.workers is not None:
        cfg["selfplay_workers"] = max(0, int(args.workers))

    #print("[CLI] Config overrides:", {k: cfg[k] for k in ["num_simulations","batch_size","log_dir","enable_tensorboard","device"] if k in cfg})
    trainer = Trainer(config=cfg)
    trainer.setup()
    # 非並行モードは廃止し、常に並行モードで実行
    total_eps = args.total_episodes if args.total_episodes is not None else args.episodes
    summary = trainer.train_concurrent(
        total_episodes=int(total_eps),
        workers=args.workers if args.workers is not None else None,
        updates_per_iter=int(args.concurrent_updates_per_iter),
        queue_maxsize=int(args.concurrent_queue_size),
    )
    # 並行モードの終了メッセージ（実績値）
    try:
        print(f"[INFO] concurrent run finished total_episodes={summary.get('episodes')} train_updates={summary.get('train_updates')}")
    except Exception:
        pass
    #print("[INFO] checkpoints ->", cfg.get("checkpoint_path"))
    #print("[INFO] replay ->", cfg.get("replay_path"))
