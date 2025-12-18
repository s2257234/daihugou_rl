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
        except Exception:
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
        )
    else:
        model = PolicyValueNet(
            max_policy_size=config.get("max_policy_size", 128),
            hidden_size=config.get("hidden_size", 128),
            num_players=config.get("num_players", 4),
            device=device,
            use_full_features=False,
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
            )
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
    return AgentEnvBundle(model=model, agents=agents, env=env, device=device, loaded_from_checkpoint=loaded)


__all__ = [
    "AgentEnvBundle",
    "resolve_device",
    "create_env_and_agents",
]
