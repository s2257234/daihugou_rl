"""Non-parallel self-play data generator.

目的:
  並行学習を行わず、自己対局 (self-play) のみを複数ワーカーで実行し
  得られたラベル付きサンプルを 1 つのジョブリブファイルへ保存する。

要件 (会話要約より):
  - 毎回の実行で data/ 以下に新しい自己対局データファイルを生成 (既存ファイルは保持)。
  - logs/ への既存 train_updates.csv / events.log は継続して追記 (clear しない)。
  - 累積エピソード数を meta.json で追跡し、再実行で継続。 (data/meta.json)
  - checkpoint の最新モデル (policy_value_latest.pt) を読み込んで対局に使用。
  - workers>1 の並列自己対局に対応。
  - 追加計測ログ ([perf-ep] 等) は workers 内部ロガー経由で既存ロジックを保持。
  - サンプルへ train/val split を付与 (config.val_split_ratio)。

設計メモ:
  SelfplayDaemonWorker は flush 時に sample_queue へ各エージェントの確定サンプルを投入する。
  現在の設定では worker_zero_buffer=True なので _episode_confirmed_samples から直接送信される。
  サンプルにはまだ 'split' が無いので drain 時に付与する。

出力ファイルフォーマット (joblib.dump):
  {
	'meta': {...},            # ラン情報 (下記)
	'samples': [ {..}, .. ]   # ラベル付き学習サンプル (value, pi_q 等)
  }
  meta: {
	 'episodes_this_run': int,
	 'episodes_cumulative': int,
	 'sample_count': int,
	 'generated_at': <unix_ts>,
	 'model_version': <int|None>,
	 'workers': int,
	 'config_md5': <str>,
	 'checkpoint_path': <str>,
	 'val_split_ratio': <float>,
  }

CLI 引数:
  --episodes N               生成するエピソード総数
  --workers W                ワーカープロセス数 (未指定は config.selfplay_workers / 最低1)
  --data-dir PATH            出力ディレクトリ (既定: data)
  --log-dir PATH             ログディレクトリ (既定: logs) ※ clear しない
  --checkpoint-dir PATH      チェックポイントディレクトリ (既定: checkpoints)
  --model-path PATH          最新モデルパス (既定: <checkpoint-dir>/policy_value_latest.pt)
  --config-json PATH         追加/上書き設定 JSON ファイル
  --seed INT                 シード上書き
  --device STR               デバイス指定 (cpu/cuda/cuda:0)
  --no-clear-logs            logger 初期化時に clear_existing=False を強制

注意:
  学習 (train_step) はこのスクリプトでは実行しない。生成されたデータファイルは
  後続 non_parallel_trainer.py (未実装) で読み込んで学習に用いる想定。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
import hashlib
import multiprocessing as mp
from typing import Any, Dict, List
import threading as _th
import queue as _q
import torch

import joblib
from collections import deque

# Ensure project root is on sys.path when run as a script
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJ_ROOT not in sys.path:
	sys.path.insert(0, _PROJ_ROOT)

from agents.config import ALPHA_ZERO_CONFIG
from utils.logger import TrainingLogger
from trainer.workers import selfplay_daemon_worker_entry
from agents.replay_buffer import store_replay_sample


def _init_duplicate_filter_state(cfg: Dict[str, Any]) -> Dict[str, Any] | None:
	"""Initialize state for save-stage duplicate filtering.

	Current policy (forced_pass-friendly):
	- Compress forced_pass samples of identical state into 1.
	- Collapse consecutive forced_pass runs by keeping the newest sample.

	Non-forced_pass samples are always kept.
	"""
	try:
		if not bool(cfg.get('enable_duplicate_filter', False)):
			return None
	except Exception:
		return None
	return {
		'enabled': True,
		'kept': [],
		'raw_count': 0,
		'removed': 0,
		'pass_only_raw': 0,
		'pass_only_kept': 0,
		'pass_only_removed_same_state': 0,
		'pass_only_removed_consecutive': 0,
		'unique_forced_states': set(),
		'_last_kept_is_forced_pass': False,
		'_last_kept_forced_key': None,
		'_last_kept_forced_player_id': None,
	}


def _duplicate_filter_add_samples(state: Dict[str, Any], samples: List[Dict[str, Any]]) -> None:
	"""Update duplicate-filter state with a batch of samples.

	Filtering is applied only to forced_pass samples:
	- Same-state forced_pass -> keep 1.
	- Consecutive forced_pass -> keep newest.
	"""
	if not state or not samples:
		return
	import json as _json
	import numpy as _np
	kept: list = state['kept']
	unique_forced_states: set = state.get('unique_forced_states') or set()
	state['unique_forced_states'] = unique_forced_states

	def _is_forced_pass_sample(s: Any) -> bool:
		if not isinstance(s, dict):
			return False
		la = s.get('legal_actions')
		if not isinstance(la, list) or len(la) == 0:
			return False
		try:
			return all(a is None for a in la)
		except Exception:
			return False

	def _state_key_for_forced_pass(s: Dict[str, Any]) -> str:
		"""Best-effort state key for forced_pass dedup.

		Prefer NN input tensor bytes (state['full_input']). If missing, fall back to a
		stable JSON signature of a small subset of state keys.
		"""
		st = s.get('state')
		pid = None
		try:
			pid = s.get('player_id')
			if pid is None and isinstance(st, dict):
				pid = st.get('self_player_id')
		except Exception:
			pid = None
		if isinstance(st, dict):
			fi = st.get('full_input')
			if fi is not None:
				try:
					arr = _np.asarray(fi)
					arr = arr.reshape(-1)
					# float16 for stability/compactness
					arr16 = arr.astype(_np.float16, copy=False)
					ver = st.get('full_input_version')
					dim = int(arr16.size)
					prefix = f"pid={pid}|v={ver}|dim={dim}|".encode('utf-8', 'ignore')
					b = prefix + arr16.tobytes(order='C')
					return hashlib.md5(b).hexdigest()
				except Exception:
					pass
			# fallback: pick a small stable subset if present
			try:
				keys = [
					'self_hand_indices',
					'field_card_indices',
					'pass_flags',
					'revolution',
					'history',
					'turn',
				]
				mini = {k: st.get(k) for k in keys if k in st}
				if pid is not None:
					mini['player_id'] = pid
				js = _json.dumps(mini, sort_keys=True, ensure_ascii=False, default=str)
				return hashlib.md5(js.encode('utf-8', 'ignore')).hexdigest()
			except Exception:
				pass
		try:
			rep = repr(st)
		except Exception:
			rep = 'NA'
		return hashlib.md5(rep.encode('utf-8', 'ignore')).hexdigest()

	for s in samples:
		state['raw_count'] += 1
		try:
			if not _is_forced_pass_sample(s):
				kept.append(s)
				state['_last_kept_is_forced_pass'] = False
				state['_last_kept_forced_key'] = None
				continue
			state['pass_only_raw'] += 1
			k = _state_key_for_forced_pass(s)
			last_forced = bool(state.get('_last_kept_is_forced_pass', False))
			last_key = state.get('_last_kept_forced_key')
			cur_pid = None
			try:
				cur_pid = s.get('player_id')
				if cur_pid is None:
					cur_pid = (s.get('state') or {}).get('self_player_id')
			except Exception:
				cur_pid = None
			last_pid = state.get('_last_kept_forced_player_id')
			# consecutive forced_pass: keep newest
			if last_forced and kept and (cur_pid == last_pid):
				# If current forced_pass state already exists elsewhere (excluding the last
				# element we might replace), skip it.
				if (k in unique_forced_states) and (k != last_key):
					state['removed'] += 1
					state['pass_only_removed_same_state'] += 1
					continue
				# Replace last kept forced_pass with current one
				try:
					kept[-1] = s
				except Exception:
					# if kept[-1] fails for any reason, fall back to append
					kept.append(s)
				# accounting: we dropped the older forced_pass
				state['removed'] += 1
				state['pass_only_removed_consecutive'] += 1
				# update state-key set if key changed
				try:
					if last_key and (last_key in unique_forced_states) and (k != last_key):
						unique_forced_states.discard(last_key)
					unique_forced_states.add(k)
				except Exception:
					pass
				state['_last_kept_is_forced_pass'] = True
				state['_last_kept_forced_key'] = k
				state['_last_kept_forced_player_id'] = cur_pid
				continue

			# non-consecutive forced_pass: same-state compression
			if k in unique_forced_states:
				state['removed'] += 1
				state['pass_only_removed_same_state'] += 1
				continue
			kept.append(s)
			unique_forced_states.add(k)
			state['pass_only_kept'] += 1
			state['_last_kept_is_forced_pass'] = True
			state['_last_kept_forced_key'] = k
			state['_last_kept_forced_player_id'] = cur_pid
		except Exception:
			# Any failure -> keep sample (fail-open)
			kept.append(s)
			state['_last_kept_is_forced_pass'] = False
			state['_last_kept_forced_key'] = None
			state['_last_kept_forced_player_id'] = None


def _load_config(base: Dict[str, Any], path: str | None) -> Dict[str, Any]:
	cfg = dict(base)
	if path and os.path.isfile(path):
		try:
			with open(path, 'r', encoding='utf-8') as f:
				user = json.load(f)
			if isinstance(user, dict):
				cfg.update(user)
		except Exception as e:
			print(f"[WARN] failed to load config json '{path}': {e}")
	return cfg


def _resolve_device(dev: str | None) -> str:
	if not dev or dev == 'auto':
		return 'cuda' if (torch.cuda.is_available() and getattr(torch.version, 'cuda', None)) else 'cpu'
	if isinstance(dev, str) and dev.startswith('cuda'):
		if (not torch.cuda.is_available()) or (getattr(torch.version, 'cuda', None) is None):
			print('[WARN] requested CUDA but torch not compiled with CUDA -> fallback to cpu')
			return 'cpu'
	return dev


def _read_model_version(checkpoint_dir: str) -> int | None:
	meta_path = os.path.join(checkpoint_dir, 'metadata.json')
	if not os.path.isfile(meta_path):
		return None
	try:
		with open(meta_path, 'r', encoding='utf-8') as f:
			meta = json.load(f)
		mv = meta.get('model_version')
		if isinstance(mv, int):
			return mv
	except Exception:
		pass
	return None


VALUE_U8_NONE = 0xFF  # Legacy constant for backwards compatibility


def _has_value_label(sample: Dict[str, Any]) -> bool:
	"""Returns True when the sample includes a valid value label.
	
	Primary check is for raw float 'value' field.
	For backwards compatibility, also accepts legacy 'value_u8' field.
	"""
	if not isinstance(sample, dict):
		return False
	# Primary: raw float value
	if sample.get('value') is not None:
		return True
	# Backwards compatibility: check legacy value_u8
	value_u8 = sample.get('value_u8')
	return isinstance(value_u8, int) and 0 <= value_u8 < VALUE_U8_NONE


def _drain_samples(sample_queue, max_batch: int | None, val_ratio: float) -> List[Dict[str, Any]]:
	out: List[Dict[str, Any]] = []
	import random as _r
	while True:
		if max_batch is not None and len(out) >= max_batch:
			break
		try:
			s = sample_queue.get_nowait()
		except Exception:
			break
		if not isinstance(s, dict):
			continue
		# value が確定していないサンプルは学習対象外なので skip
		if not _has_value_label(s):
			continue
		if s.get('split') is None:
			try:
				s['split'] = 'val' if (_r.random() < val_ratio) else 'train'
			except Exception:
				s['split'] = 'train'
		out.append(s)
	return out


def run_self_play(cfg: Dict[str, Any], episodes: int, workers: int, model_path: str, data_dir: str, log_dir: str, checkpoint_dir: str) -> str:
	assert episodes > 0
	os.makedirs(data_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(checkpoint_dir, exist_ok=True)

	# logger: clear_existing を抑止してログ継続
	logger = TrainingLogger(
		log_dir=log_dir,
		use_tensorboard=cfg.get('enable_tensorboard', True),
		clear_existing=False,  # 常に追記継続
		log_mcts_samples=not cfg.get('disable_mcts_log', False),
		config=cfg,
	)
	# Ensure at least one visible startup line is written so short runs produce a visible events.log
	try:
		logger.log_text(f"[status-np] started episodes={episodes} workers={workers} model={os.path.basename(model_path)} data_dir={data_dir} checkpoint_dir={checkpoint_dir}")
		# force immediate flush so very short runs still leave the line on disk
		try:
			logger.flush_buffers(force=True)
		except Exception:
			pass
	except Exception:
		pass

	# 累積メタ読み込み (data/meta.json)
	meta_path = os.path.join(data_dir, 'meta.json')
	cumulative_eps = 0
	meta_obj = {}
	train_updates_cum = 0
	replay_size_cached = 0
	if os.path.isfile(meta_path):
		try:
			with open(meta_path, 'r', encoding='utf-8') as f:
				meta_obj = json.load(f)
			cumulative_eps = int(meta_obj.get('episodes_cumulative', 0) or 0)
			train_updates_cum = int(meta_obj.get('train_updates_cumulative', 0) or 0)
			replay_size_cached = int(meta_obj.get('replay_size', 0) or 0)
		except Exception:
			pass

	# ワーカー起動
	workers = max(1, int(workers))
	use_thread_mode = (workers == 1) or bool(cfg.get('force_serial_self_play', False))
	procs: List[mp.Process] = []
	threads: List[_th.Thread] = []
	control_queues = []
	if use_thread_mode:
		# スレッド + queue で実行（Windows spawn のpickle問題を完全回避）
		sample_queue = _q.Queue(maxsize=max(1000, int(cfg.get('selfplay_queue_maxsize', 50000) or 50000)))
		event_queue = _q.Queue(maxsize=10000)
		stop_event = _th.Event()
	else:
		ctx = mp.get_context('spawn')
		sample_queue = ctx.Queue(maxsize=max(1000, int(cfg.get('selfplay_queue_maxsize', 50000) or 50000)))
		event_queue = ctx.Queue(maxsize=10000)
		stop_event = ctx.Event()

	# モデル存在チェック (なければ警告表示。SelfplayDaemonWorker 内部で新規モデルを生成するフォールバックあり)
	if not os.path.isfile(model_path):
		print(f"[WARN] model path not found: {model_path} -> new model will be initialized")

	# 軽量設定強制 (共有リプレイ無効 / worker_zero_buffer 有効)
	cfg_local = dict(cfg)
	cfg_local['use_shared_replay'] = False
	cfg_local['worker_zero_buffer'] = True
	cfg_local['clear_logs_on_start'] = False  # 念押し
	# Windows spawn のpickle対策: 子プロセスへは JSON 文字列で渡す
	try:
		cfg_local_json = json.dumps(cfg_local, ensure_ascii=False)
	except Exception:
		cfg_local_json = json.dumps({})

	if use_thread_mode:
		# 単一ワーカーをスレッドで起動
		cq = _q.Queue(maxsize=5)
		control_queues.append(cq)
		def _thread_target():
			# スレッドでは辞書のまま渡す（pickle不要）
			selfplay_daemon_worker_entry(0, cfg_local, model_path, sample_queue, event_queue, stop_event, cq, None, None)
		th = _th.Thread(target=_thread_target, name="selfplay-worker-0", daemon=True)
		th.start()
		threads.append(th)
	else:
		for wid in range(workers):
			cq = ctx.Queue(maxsize=5)
			control_queues.append(cq)
			p = ctx.Process(
				target=selfplay_daemon_worker_entry,
				args=(wid, cfg_local_json, model_path, sample_queue, event_queue, stop_event, cq, None, None),
				daemon=False,
			)
			try:
				p.start()
			except Exception as e:
				print(f"[ERROR] failed to start worker wid={wid}: {e}")
				continue
			procs.append(p)

	target_eps = int(episodes)
	ep_done = 0
	# 累積表示用: この実行開始前までの累積エピソードを保持
	start_cumulative_eps = cumulative_eps
	drained_samples: List[Dict[str, Any]] = []
	val_ratio = float(cfg.get('val_split_ratio', 0.0) or 0.0)
	dup_state = _init_duplicate_filter_state(cfg)
	if dup_state is not None:
		# keep list is owned by dup_state; bind drained_samples to it so len(drained_samples)
		# naturally becomes the post-dedup count for status-np
		drained_samples = dup_state['kept']
	# ステータス連打抑制用: 直近ログ出力した ep 値と直近の drained サンプル数
	last_status_logged_ep = -1
	last_status_logged_samples = 0

	last_progress_len = 0
	use_bar = bool(cfg.get('use_progress_bar', False)) and bool(cfg.get('minimal_progress', False))

	# worker 終了検知: 想定外の早期終了で無限待ちしないための保険
	alive_workers = int(workers)

	start_ts = time.time()
	try:
		while ep_done < target_eps:
			# イベント処理
			try:
				evt, val = event_queue.get(timeout=0.25)
			except Exception:
				evt = None
			if evt == 'ep_done':
				ep_done += int(val)
				# 進捗バー (簡易)
				if use_bar:
					bar_w = int(cfg.get('progress_bar_width', 40) or 40)
					done = min(ep_done, target_eps)
					pct = done / target_eps
					filled = int(bar_w * pct)
					bar = '[' + '#' * filled + '-' * (bar_w - filled) + ']'
					line = f"[SELFPLAY(np)] {bar} {done}/{target_eps}"
					pad = max(0, last_progress_len - len(line))
					print(line + ' ' * pad, end='\r' if done < target_eps else '\n', flush=True)
					last_progress_len = len(line)
			elif evt == 'perf_ep':  # forward/game time 計測行は logger 内で処理されるためここでは何もしない
				pass
			elif evt == 'hb':
				pass
			elif evt == 'worker_exit':
				try:
					alive_workers = max(0, alive_workers - 1)
				except Exception:
					alive_workers = 0
				# 全workerが終了していて、まだ目標ep未達のときは安全に打ち切り保存へ
				if alive_workers <= 0 and ep_done < target_eps:
					if logger:
						try:
							logger.log_text(f"[WARN] all workers exited before reaching target episodes (ep_done={ep_done}/{target_eps}). proceeding to finalize with partial data.")
						except Exception:
							pass
					break
			# サンプル drain
			new_samples = _drain_samples(sample_queue, max_batch=2000, val_ratio=val_ratio)
			if new_samples:
				if dup_state is not None:
					_duplicate_filter_add_samples(dup_state, new_samples)
				else:
					drained_samples.extend(new_samples)
			# 進捗ステータス: 指定間隔かつ同じ ep で一度のみ、サンプル増分がある場合のみ出力
			# 既定のログ間隔を 250 エピソードへ（設定 measure_log_every_episodes で上書き可能）
			log_interval = max(1, int(cfg.get('measure_log_every_episodes', 250) or 250))
			if (
				logger
				and ep_done > 0
				and (ep_done % log_interval == 0)
				and ep_done != last_status_logged_ep
				and len(drained_samples) > last_status_logged_samples
			):
				# 累積エピソード値 (前回まで + 今回進捗) を ep_done として出力
				cum_ep_done = start_cumulative_eps + ep_done
				try:
					logger.log_text(
						f"[status-np] ep_done={cum_ep_done} step={train_updates_cum} new_since_train={len(drained_samples)} replay_size={replay_size_cached}"
					)
					last_status_logged_ep = ep_done
					last_status_logged_samples = len(drained_samples)
				except Exception:
					pass
		# 完了後: フラッシュ要求送信
		stop_event.set()
		for cq in control_queues:
			try:
				cq.put('FLUSH_AND_EXIT', block=False)
			except Exception:
				pass
	finally:
		# 最終 drain
		final_samples = _drain_samples(sample_queue, max_batch=None, val_ratio=val_ratio)
		if final_samples:
			if dup_state is not None:
				_duplicate_filter_add_samples(dup_state, final_samples)
			else:
				drained_samples.extend(final_samples)
		if use_thread_mode:
			# スレッドは stop 指示後に短い待機
			for th in threads:
				try:
					th.join(timeout=5)
				except Exception:
					pass
		else:
			# join workers (with timeout) then force terminate lingering ones
			for p in procs:
				p.join(timeout=10)
			# 強制終了フェーズ（長時間ぶら下がり防止）
			alive_pids = []
			for p in procs:
				try:
					if p.is_alive():
						alive_pids.append(p.pid)
						p.terminate()
				except Exception:
					pass
			if alive_pids:
				print(f"[WARN] workers pids={alive_pids} still alive after join timeout; terminating.")
			for p in procs:
				try:
					if p.is_alive():
						p.join(timeout=3)
				except Exception:
					pass

	elapsed = time.time() - start_ts

	# For reporting: raw is pre-dedup drained count, kept is len(drained_samples).
	if dup_state is not None:
		raw_sample_count = int(dup_state.get('raw_count', 0) or 0)
		dedup_removed = int(dup_state.get('removed', 0) or 0)
		dedup_unique_sigs = int(len(dup_state.get('unique_forced_states', set()) or set()))
	else:
		raw_sample_count = len(drained_samples)
		dedup_removed = 0
		dedup_unique_sigs = 0

	# model_version 取得
	model_version = _read_model_version(checkpoint_dir)
	# config ハッシュ
	cfg_hash = hashlib.md5(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()

	# メタ更新 (累積) — 既存 meta 内の train_updates_cumulative / replay_size を保持して上書きしない
	cumulative_eps += ep_done
	new_meta = {
		'episodes_cumulative': cumulative_eps,
		'episodes_this_run': ep_done,
		'sample_count': len(drained_samples),
		'sample_count_raw': raw_sample_count,
		'duplicate_filter_removed': int(dedup_removed),
		'duplicate_filter_unique_sigs': int(dedup_unique_sigs),
		'generated_at': time.time(),
		'model_version': model_version,
		'workers': workers,
		'config_md5': cfg_hash,
		'checkpoint_path': model_path,
		'val_split_ratio': val_ratio,
		# 既存保持フィールド (trainer が継続カウンタを失わないようにする)
		'train_updates_cumulative': train_updates_cum,
		'replay_size': replay_size_cached,
	}
	try:
		if not cfg.get('disable_data_writes', False):
			with open(meta_path, 'w', encoding='utf-8') as f:
				json.dump(new_meta, f, ensure_ascii=False, indent=2)
		else:
			print(f"[INFO] data writes disabled by config; skipping meta.json -> {meta_path}")
	except Exception as e:
		print(f"[WARN] failed to write meta.json: {e}")

	# 出力ファイル名
	ts_tag = time.strftime('%Y%m%d_%H%M%S')
	fname = f"selfplay_ep{ep_done}_mv{model_version if model_version is not None else 'NA'}_{ts_tag}.joblib"
	out_path = os.path.join(data_dir, fname)
	payload = {
		'meta': new_meta,
		'samples': drained_samples,
	}

	# Emit a concise one-line summary per self-play run.
	try:
		if bool(cfg.get('enable_duplicate_filter', False)) and raw_sample_count > 0 and dup_state is not None:
			forced_raw = int(dup_state.get('pass_only_raw', 0) or 0)
			forced_kept = int(dup_state.get('pass_only_kept', 0) or 0)
			removed_same = int(dup_state.get('pass_only_removed_same_state', 0) or 0)
			removed_cons = int(dup_state.get('pass_only_removed_consecutive', 0) or 0)
			uniq_forced = int(len(dup_state.get('unique_forced_states') or set()))
			summary_line = (
				f"[DUP_SUM] episodes={ep_done} samples={raw_sample_count} kept={len(drained_samples)} removed={dedup_removed} "
				f"pass_only={forced_raw} pass_only_kept={forced_kept} "
				f"pass_only_removed_same_state={removed_same} pass_only_removed_consecutive={removed_cons} "
				f"unique_forced_states={uniq_forced} reason=forced_pass_state_key+consecutive"
			)
			sum_path = os.path.join(log_dir, 'duplicate_skipped_summary.log')
			with open(sum_path, 'a', encoding='utf-8') as sf:
				sf.write(summary_line + '\n')
	except Exception:
		pass
	# 圧縮レベル:
	# 直列モード (non-parallel) は memmap 利用とI/O高速化を優先し compress=0 を強制。
	# 並列モードと区別するための設定キー selfplay_joblib_compress があってもここでは無視。
	# 必要なら config-json で selfplay_force_compress>=0 を指定して上書き可能。
	force_c = cfg.get('selfplay_force_compress', None)
	if force_c is not None:
		try:
			compress_lv = int(force_c)
		except Exception:
			compress_lv = 0
	else:
		compress_lv = 0  # non-parallel 強制無圧縮
	try:
		if not cfg.get('disable_data_writes', False):
			tmp = out_path + '.tmp'
			joblib.dump(payload, tmp, compress=compress_lv)
			os.replace(tmp, out_path)
		else:
			out_path = None
			print(f"[INFO] data writes disabled by config; skipping self-play output file creation")
	except Exception as e:
		print(f"[ERROR] failed to save data file {out_path}: {e}")
	# ログへ完了サマリー
	# 最終サマリータグは既存 status-np シリーズで十分なため削除 (冗長抑制)
	if logger and hasattr(logger, 'flush_buffers'):
		try:
			logger.flush_buffers(force=True)
		except Exception:
			pass
	print(f"[INFO] self-play finished episodes={ep_done} samples={len(drained_samples)} elapsed={elapsed:.1f}s -> {out_path}")
	return out_path


def main():
	parser = argparse.ArgumentParser(description="Non-parallel self-play data generator")
	parser.add_argument('--episodes', type=int, default=500, help='生成する自己対局エピソード数')
	parser.add_argument('--workers', type=int, default=16, help='ワーカー数 (未指定は設定値 selfplay_workers)')
	parser.add_argument('--data-dir', type=str, default='data', help='自己対局データ出力ディレクトリ')
	parser.add_argument('--log-dir', type=str, default='logs', help='ログディレクトリ (継続追記)')
	parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='チェックポイントディレクトリ')
	parser.add_argument('--model-path', type=str, default=None, help='最新モデルパス (未指定は <checkpoint-dir>/policy_value_latest.pt)')
	parser.add_argument('--config-json', type=str, default=None, help='設定上書き JSON パス')
	parser.add_argument('--seed', type=int, default=None, help='乱数シード上書き')
	parser.add_argument('--device', type=str, default=None, help='デバイス指定 (cpu/cuda)')
	parser.add_argument('--no-clear-logs', action='store_true', help='ログ初期化時の既存削除を無効化')
	args = parser.parse_args()

	base_cfg = _load_config(ALPHA_ZERO_CONFIG, args.config_json)
	# override
	if args.seed is not None:
		base_cfg['seed'] = int(args.seed)
	if args.device is not None:
		base_cfg['device'] = args.device
	if args.no_clear_logs:
		base_cfg['clear_logs_on_start'] = False
	else:
		# 強制的に False (非並列モードは過去ログ維持) — ユーザがあえて消したい場合は config-json で True
		base_cfg['clear_logs_on_start'] = False
	# 進捗バー簡易化: デフォルト最小進捗 False ならバー OFF
	base_cfg.setdefault('minimal_progress', True)
	base_cfg.setdefault('use_progress_bar', True)
	# 学習関連は無効化方向へ (安全にメモリ削減)
	base_cfg['use_shared_replay'] = False
	base_cfg['purge_replay_after_each_update'] = False
	base_cfg['purge_replay_after_checkpoint'] = False

	workers = args.workers if args.workers is not None else int(base_cfg.get('selfplay_workers', 1) or 1)
	model_path = args.model_path if args.model_path else os.path.join(args.checkpoint_dir, 'policy_value_latest.pt')
	# ensure device resolved for potential model loads (SelfplayDaemonWorker handles internally)
	base_cfg['device'] = _resolve_device(base_cfg.get('device'))

	run_self_play(
		cfg=base_cfg,
		episodes=int(args.episodes),
		workers=int(workers),
		model_path=model_path,
		data_dir=args.data_dir,
		log_dir=args.log_dir,
		checkpoint_dir=args.checkpoint_dir,
	)


if __name__ == '__main__':
	# Windows spawn 安定化: freeze_support 呼び出しで __main__ を明確化
	try:
		import multiprocessing as _mp
		_mp.freeze_support()
	except Exception:
		pass
	main()

