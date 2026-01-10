from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os

from agents.drl_agent import AlphaZeroAgent
from agents.models import PolicyValueNet
from game.environment import DaifugoSimpleEnv


@dataclass
class AgentEnvBundle:
    model: PolicyValueNet
    agents: List[AlphaZeroAgent]
    env: DaifugoSimpleEnv
    device: str
    loaded_from_checkpoint: bool


def resolve_device(config: Dict[str, Any], *, context: str = "main") -> str:
    device_cfg = config.get("device", "auto")
    if context == "worker" and bool(config.get("force_worker_cpu", True)):
        return "cpu"
    if device_cfg == "auto":
        try:
            import torch  # type: ignore
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return str(device_cfg)


def _load_or_build_model(config: Dict[str, Any], *, device: str, model_path: Optional[str]) -> Tuple[PolicyValueNet, bool]:
    use_full = bool(config.get("use_full_features", False))
    loaded = False
    if model_path and os.path.isfile(model_path):
        try:
            model = PolicyValueNet.load(model_path, map_location=device)
            loaded = True
            return model, loaded
        except Exception as e:
            # 古いチェックポイント互換性エラーを詳細にログ
            print(f"[WARN] Failed to load checkpoint '{model_path}': {e}")
            print(f"[INFO] Falling back to new model creation")
            # フォールバックで新規作成へ
            pass
    # 新規作成
    if use_full:
        # 新仕様に合わせたデフォルト full_feature_dim を使う。
        # models.py のレイアウトに基づく期待式: full_feature_dim = 72*N + 73
        try:
            N = int(config.get("num_players", 4))
            # New layout: full_feature_dim = 73*N + 74
            default_full_dim = 73 * N + 74
        except Exception:
            default_full_dim = 366
        model = PolicyValueNet(
            max_policy_size=config.get("max_policy_size", 128),
            hidden_size=config.get("hidden_size", 128),
            num_players=config.get("num_players", 4),
            device=device,
            use_full_features=True,
            # 初期は新仕様に合わせた次元、後でプローブで再構築（必要なら）
            full_feature_dim=default_full_dim,
            context_out_dim=config.get("hidden_size", 128),  # hidden_sizeに合わせる（過学習防止）
            signal_noise_std=0.0,  # Signal Augmentationを無効化（勾配爆発対策）
        )
    else:
        # use_full_features=False は廃止されているが、互換性のため残している
        model = PolicyValueNet(
            max_policy_size=config.get("max_policy_size", 128),
            hidden_size=config.get("hidden_size", 128),
            num_players=config.get("num_players", 4),
            device=device,
            use_full_features=False,
            signal_noise_std=0.0,  # Signal Augmentationを無効化（勾配爆発対策）
        )
    return model, loaded


def _attach_agent_context(
    agents: List[AlphaZeroAgent],
    env: DaifugoSimpleEnv,
    *,
    logger,
    shared_replay,
    context: str,
    worker_zero_buffer: bool,
    request_q=None,
    response_q=None,
    worker_id: Optional[int] = None,
):
    env.agents = agents
    for ag in agents:
        if hasattr(ag, "set_env_ref"):
            ag.set_env_ref(env)
        # ロガーは main のみ接続（worker 内はノイズ防止で None）
        try:
            ag.logger = logger if context == "main" else None
        except Exception:
            pass
        if context == "main":
            if shared_replay is not None:
                ag.replay_buffer = shared_replay
                # non_parallel_trainer では常に shared_replay を渡すため、_use_shared を True に設定
                try:
                    ag._use_shared = True
                except Exception:
                    pass
        else:
            # worker: 共有RBは使わずローカル保持（ゼロバッファ運用が有効なら None）
            try:
                ag._use_shared = False
                ag.replay_buffer = None if worker_zero_buffer else []
            except Exception:
                pass
            # 中央推論（あれば）
            if request_q is not None and response_q is not None:
                try:
                    ag._remote_request_q = request_q
                    ag._remote_response_q = response_q
                    ag._remote_worker_id = int(worker_id) if worker_id is not None else -1
                    ag._remote_req_counter = 0
                except Exception:
                    pass


def _maybe_rebuild_full_model(config: Dict[str, Any], model: PolicyValueNet, agents: List[AlphaZeroAgent], env: DaifugoSimpleEnv) -> PolicyValueNet:
    use_full = bool(config.get("use_full_features", False))
    if not use_full:
        return model
    try:
        env.reset()
        probe = agents[0]._extract_state(env)
        full_dim = probe.get("full_input_dim")
        cur_full = bool(getattr(model, "use_full_features", False))
        # 既存モデルが持つ full_feature_dim を直接参照する（models.PolicyValueNet は属性を持つ）
        model_full = getattr(model, "full_feature_dim", None)
        # 再構築が必要か: probe に次元があり、モデル側が未フル or 次元が一致しない場合は再構築
        need_rebuild = False
        if full_dim is not None:
            try:
                full_dim_i = int(full_dim)
            except Exception:
                full_dim_i = None
            if full_dim_i is not None:
                if (not cur_full) or (model_full is None) or (int(model_full) != full_dim_i):
                    need_rebuild = True
        if need_rebuild:
            device = next(model.parameters()).device if hasattr(model, "parameters") else None
            device_str = str(device) if device is not None else resolve_device(config, context="main")
            new_model = PolicyValueNet(
                max_policy_size=config.get("max_policy_size", 128),
                hidden_size=config.get("hidden_size", 128),
                num_players=config.get("num_players", 4),
                device=device_str,
                use_full_features=True,
                full_feature_dim=int(full_dim_i),
                context_out_dim=config.get("hidden_size", 128),  # hidden_sizeに合わせる（過学習防止）
                signal_noise_std=0.0,  # Signal Augmentationを無効化（勾配爆発対策）
            )
            
            # 古いモデルのパラメータを可能な限りコピー
            try:
                old_state_dict = model.state_dict()
                new_state_dict = new_model.state_dict()
                # キーが一致するパラメータをコピー
                copied_count = 0
                skipped_count = 0
                zero_params_after_copy = []
                
                for key in new_state_dict.keys():
                    if key in old_state_dict:
                        old_param = old_state_dict[key]
                        new_param = new_state_dict[key]
                        # 形状が一致する場合のみコピー
                        if old_param.shape == new_param.shape:
                            # デバイスを統一（CPUに移動してからコピー）
                            if old_param.device != new_param.device:
                                old_param = old_param.cpu()
                            new_state_dict[key] = old_param.clone() if hasattr(old_param, 'clone') else old_param
                            copied_count += 1
                        else:
                            skipped_count += 1
                
                # load_state_dictを実行
                missing_keys, unexpected_keys = new_model.load_state_dict(new_state_dict, strict=False)
                
                # コピー後のパラメータを検証（重要なパラメータがゼロでないことを確認）
                critical_params = [
                    'self_encoder.0.weight',
                    'context_encoder.0.weight',
                    'backbone_proj.weight',
                    'backbone.0.lin1.weight',
                    'backbone.0.lin2.weight',
                    'policy_head.0.weight',
                    'value_head.0.weight',
                ]
                
                for key in critical_params:
                    if key in new_model.state_dict():
                        param = new_model.state_dict()[key]
                        if hasattr(param, 'abs'):
                            abs_max = param.abs().max().item()
                            if abs_max < 1e-8:
                                zero_params_after_copy.append(key)
                
                if zero_params_after_copy:
                    print(f"[WARN] _maybe_rebuild_full_model: After parameter copy, zero parameters found: {zero_params_after_copy}")
                    # ゼロパラメータが見つかった場合、元のパラメータを再確認
                    for key in zero_params_after_copy:
                        if key in old_state_dict:
                            old_abs_max = old_state_dict[key].abs().max().item()
                            print(f"  [INFO] Original {key}: max={old_abs_max:.6f}")
                            if old_abs_max > 1e-8:
                                # 元のパラメータは正常なので、再コピーを試みる
                                try:
                                    old_param = old_state_dict[key]
                                    target_param = new_model.state_dict()[key]
                                    # デバイスを統一
                                    if old_param.device != target_param.device:
                                        old_param = old_param.to(target_param.device)
                                    # copy_を使用して確実にコピー
                                    import torch
                                    with torch.no_grad():
                                        target_param.copy_(old_param)
                                    # コピー後の検証
                                    copied_abs_max = target_param.abs().max().item()
                                    if copied_abs_max > 1e-8:
                                        print(f"  [INFO] Re-copied {key} successfully (max={copied_abs_max:.6f})")
                                    else:
                                        print(f"  [ERROR] Re-copy failed: {key} is still zero after copy_")
                                except Exception as re_copy_e:
                                    print(f"  [ERROR] Failed to re-copy {key}: {re_copy_e}")
                                    import traceback
                                    traceback.print_exc()
                
                # デバッグ用: コピーされたパラメータ数をログ出力（オプション）
                if config.get("debug_model_rebuild", False):
                    print(f"[factory] _maybe_rebuild_full_model: copied {copied_count}/{len(new_state_dict)} parameters from old model (skipped {skipped_count} due to shape mismatch)")
            except Exception as e:
                # パラメータコピーに失敗した場合は警告のみ（初期化済みモデルで継続）
                print(f"[WARN] Failed to copy parameters from old model to new model during rebuild: {e}")
                import traceback
                traceback.print_exc()
            
            for ag in agents:
                try:
                    ag.set_model(new_model)
                except Exception:
                    pass
            return new_model
    except Exception:
        # 再構築失敗時は現モデルで継続
        pass
    return model


def create_env_and_agents(
    config: Dict[str, Any],
    *,
    context: str = "main",  # "main" | "worker"
    model_path: Optional[str] = None,
    resolved_device: Optional[str] = None,
    shared_replay=None,
    logger=None,
    worker_id: Optional[int] = None,
    request_q=None,
    response_q=None,
    worker_zero_buffer: bool = False,
) -> AgentEnvBundle:
    # デバイス決定
    device = resolved_device or resolve_device(config, context=context)
    # モデル準備
    model, loaded = _load_or_build_model(config, device=device, model_path=model_path)
    # エージェント作成
    num_players = int(config.get("num_players", 4))
    agents: List[AlphaZeroAgent] = []
    for pid in range(num_players):
        ag = AlphaZeroAgent(player_id=pid, model=model, config=dict(config))
        agents.append(ag)
    # 環境作成
    env = DaifugoSimpleEnv(num_players=num_players, agent_classes=None)
    # デバッグオプション設定
    if bool(config.get("debug_action_mismatch", False)):
        env.debug_action_mismatch = True
    # 文脈に応じて接続
    _attach_agent_context(
        agents,
        env,
        logger=logger,
        shared_replay=shared_replay,
        context=context,
        worker_zero_buffer=worker_zero_buffer,
        request_q=request_q,
        response_q=response_q,
        worker_id=worker_id,
    )
    # フル特徴量時、必要なら入力次元に合わせて再構築
    model = _maybe_rebuild_full_model(config, model, agents, env)
    # Optional: force model full feature dim override from config (useful when model expects legacy dim)
    try:
        forced = config.get('force_full_input_dim', None)
        if forced is not None:
            try:
                fd = int(forced)
                # apply to model attribute (best-effort)
                try:
                    model.full_feature_dim = fd
                except Exception:
                    pass
                # also update existing agents' model references if they hold the model
                for ag in agents:
                    try:
                        if getattr(ag, 'model', None) is not None:
                            ag.model.full_feature_dim = fd
                    except Exception:
                        pass
                print(f"[INFO] forced model.full_feature_dim to {fd} via config.force_full_input_dim")
            except Exception:
                pass
    except Exception:
        pass
    # bundle.modelとlearner.modelが同じインスタンスであることを保証
    # _maybe_rebuild_full_modelで新しいモデルが作成された場合、各エージェントのモデルは更新されているが、
    # bundle.modelも同じインスタンスを参照するようにする
    bundle = AgentEnvBundle(model=model, agents=agents, env=env, device=device, loaded_from_checkpoint=loaded)
    # 念のため、学習プレイヤーのモデルとbundle.modelが同じインスタンスであることを確認
    learning_player_id = config.get('learning_player_id', 0)
    if learning_player_id < len(agents):
        learner = agents[learning_player_id]
        if id(learner.model) != id(bundle.model):
            # 学習プレイヤーのモデルとbundle.modelが異なる場合は、bundle.modelを更新
            import warnings
            warnings.warn(
                f"[factory] bundle.model and learner.model are different instances. "
                f"Updating bundle.model to match learner.model. "
                f"bundle.model ID: {id(bundle.model)}, learner.model ID: {id(learner.model)}",
                RuntimeWarning
            )
            # bundle.modelを学習プレイヤーのモデルに更新（dataclassのため直接代入できないので、新しいBundleを作成）
            bundle = AgentEnvBundle(
                model=learner.model,
                agents=agents,
                env=env,
                device=device,
                loaded_from_checkpoint=loaded
            )
    return bundle


__all__ = [
    "AgentEnvBundle",
    "resolve_device",
    "create_env_and_agents",
]
