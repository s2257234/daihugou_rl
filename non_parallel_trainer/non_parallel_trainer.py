"""Non-parallel training script.

目的:
  - `non_parallel_self_play.py` が出力した複数の自己対局データ(joblib)を段階的に読み込み、
	リプレイバッファへ追加して学習 (train_step) を実行する。
  - 既存ログ (`logs/`) を消さず継続追記。`train_updates.csv` も行追加のみ。
  - チェックポイント (最新 + 任意世代タグ) を書き出し、optimizer / scheduler 状態を保存。
  - `data/meta.json` へ累積エピソード、学習更新回数 (train_updates_cumulative) を反映。
  - 既に取り込んだ自己対局ファイルは再読込しない (meta に consumed_files を保持)。

ファイル構造想定 (self-play 側):
  data/selfplay_epXXXX_mvY_YYYYmmdd_HHMMSS.joblib
	{ 'meta': {...}, 'samples': [ sample_dict, ... ] }

サンプル利用:
  - sample['split'] == 'train' を学習、'val' を検証セット候補に保持。
  - full feature モード時でもサンプル構造は ReplayBuffer.append へ直接渡す。

ステップ概要:
  1. 設定 / CLI 引数読み込み (ALPHA_ZERO_CONFIG 上書き)
  2. ロガー初期化 (clear_existing=False)
  3. 最新モデル / optimizer / scheduler 復元 (存在すれば)
  4. data/ 下の未消費 selfplay joblib を列挙し取り込み
  5. 指定学習更新回数だけ train_step 実行 (ポリシー/価値/手札予測損失をログ)
  6. 検証 (val_eval_every_updates が設定された場合) を定期実行
  7. チェックポイント保存 & meta.json 更新

注意:
  - 非同期 / 並列自己対局は行わない。
  - メモリ肥大抑制: max_files / max_samples_per_file オプションで取り込み量を制限可能。
  - ReplayBuffer.maxlen 超過時は内部で古いサンプルが落ちる。
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
import hashlib
import glob
import random
import gc
from typing import Any, Dict, List
import warnings
import torch
import joblib

# Ensure project root is known early (used below)
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Ensure joblib temporary folder is placed inside the project to avoid
# deletion/permission issues when joblib creates memmap temp folders.
try:
	import tempfile as _tempfile
	_joblib_temp_dir = os.path.join(_PROJ_ROOT, '.joblib_temp')
	os.makedirs(_joblib_temp_dir, exist_ok=True)
	os.environ.setdefault('JOBLIB_TEMP_FOLDER', _joblib_temp_dir)
except Exception:
	pass

from agents.drl_agent import AlphaZeroAgent
from agents.replay_buffer import ReplayBuffer
from utils.logger import TrainingLogger
from agents.config import ALPHA_ZERO_CONFIG
from agents.factory import create_env_and_agents
from agents.factory import create_env_and_agents


def _maybe_collect_gc(cfg: Dict[str, Any]):
	"""Run garbage collection if aggressive_gc is enabled."""
	if not bool(cfg.get('aggressive_gc', False)):
		return
	try:
		gc.collect()
	except Exception:
		pass


def _log_memory_usage(logger: TrainingLogger | None, cfg: Dict[str, Any], note: str):
	"""Log RSS memory usage via psutil when available."""
	try:
		import psutil
	except ImportError:
		psutil = None
	if logger is None or not bool(cfg.get('log_memory_usage', True)) or psutil is None:
		return
	# throttle memory logging by configured interval (seconds)
	try:
		interval = float(cfg.get('memory_log_interval_sec', 3600) or 3600)
	except Exception:
		interval = 3600.0
	# module-level last log timestamp
	global _LAST_MEM_LOG_TS
	try:
		last = float(globals().get('_LAST_MEM_LOG_TS', 0.0) or 0.0)
	except Exception:
		last = 0.0
	import time as _time
	now = _time.time()
	if (now - last) < max(0.0, interval):
		return
	try:
		proc = psutil.Process(os.getpid())
		rss_mb = proc.memory_info().rss / (1024 * 1024)
		logger.log_text(f"[mem] {note} rss_mb={rss_mb:.2f}")
		try:
			globals()['_LAST_MEM_LOG_TS'] = now
		except Exception:
			pass
	except Exception:
		pass

# --------------------------------------------------
# Buffer State 永続化（過学習防止のためのスライディングウィンドウ）
# --------------------------------------------------
def _load_buffer_state(data_dir: str) -> Dict[str, Any]:
	"""buffer_state.jsonから前回のファイルリストを読み込む
	
	スライディングウィンドウ方式でファイルリストを管理し、
	同じデータの繰り返し学習を防止する。
	"""
	path = os.path.join(data_dir, 'buffer_state.json')
	if os.path.isfile(path):
		try:
			with open(path, 'r', encoding='utf-8') as f:
				state = json.load(f)
			# ファイルリストのバリデーション（存在しないファイルを除外）
			active_files = state.get('active_files', [])
			valid_files = [f for f in active_files if os.path.isfile(f)]
			if len(valid_files) != len(active_files):
				state['active_files'] = valid_files
			# used_filesフィールドが存在しない場合は初期化
			if 'used_files' not in state:
				state['used_files'] = []
			return state
		except Exception:
			pass
	return {'active_files': [], 'used_files': [], 'last_updated': None, 'file_count': 0}


def _save_buffer_state(data_dir: str, active_files: List[str], used_files: List[str] | None = None, logger=None):
	"""現在のファイルリストをbuffer_state.jsonに保存
	
	次回学習時に同じファイルを再度読み込まないようにするため、
	現在のウィンドウ状態を永続化する。
	
	Args:
		active_files: 現在アクティブなファイルリスト
		used_files: 使用済みファイルリスト（Noneの場合は既存のused_filesを維持）
		logger: ログ出力用
	"""
	path = os.path.join(data_dir, 'buffer_state.json')
	# 既存のstateを読み込んでused_filesを維持
	prev_used_files = []
	try:
		if os.path.isfile(path):
			with open(path, 'r', encoding='utf-8') as f:
				prev_state = json.load(f)
				prev_used_files = prev_state.get('used_files', [])
	except Exception:
		pass
	
	# used_filesが指定されている場合は更新、そうでなければ既存を維持
	if used_files is not None:
		# 使用済みファイルリストに追加（重複を避ける）
		new_used_set = set(prev_used_files) | set(used_files)
		final_used_files = list(new_used_set)
	else:
		final_used_files = prev_used_files
	
	state = {
		'active_files': list(active_files),
		'used_files': final_used_files,
		'last_updated': time.time(),
		'file_count': len(active_files)
	}
	try:
		_atomic_write_json(path, state)
		if logger:
			logger.log_text(f"[buffer-state] saved {len(active_files)} active files, {len(final_used_files)} used files to buffer_state.json")
	except Exception as e:
		if logger:
			logger.log_text(f"[WARN] buffer_state save failed: {e}")


def _update_buffer_window(
	prev_files: List[str],
	all_files: List[str],
	window_size: int,
	used_files: List[str] | None = None,
	enable_random_selection: bool = False,
	logger=None
) -> tuple[List[str], List[str]]:
	"""スライディングウィンドウでファイルリストを更新（改善案A・B対応）
	
	Args:
		prev_files: 前回のファイルリスト（buffer_state.jsonから読み込み）
		all_files: 現在利用可能な全ファイル（ソート済み、時系列昇順）
		window_size: ウィンドウサイズ（保持するファイル数の上限）
		used_files: 使用済みファイルリスト（改善案A: これらを除外）
		enable_random_selection: 改善案B: Trueの場合はランダムに選択
		logger: ログ出力用
	
	Returns:
		(更新されたファイルリスト, 削除されたファイルリスト)
	
	Note:
		改善案A: 使用済みファイルを除外して新しいファイルセットを選択
		改善案B: enable_random_selection=Trueの場合、ランダムにファイルを選択
	"""
	import random as _r
	
	# 利用可能なファイルから使用済みファイルを除外（改善案A）
	available_files = all_files
	if used_files:
		used_set = set(used_files)
		available_files = [f for f in all_files if f not in used_set and os.path.isfile(f)]
		if logger:
			logger.log_text(f"[buffer-window] excluded {len(used_files)} used files, available={len(available_files)}")
	
	# 利用可能なファイルが不足している場合は、全ファイルから選択
	if len(available_files) < window_size:
		available_files = [f for f in all_files if os.path.isfile(f)]
		if logger:
			logger.log_text(f"[buffer-window] not enough files after exclusion, using all {len(available_files)} files")
	
	if not available_files:
		if logger:
			logger.log_text("[buffer-window] no available files")
		return [], []
	
	# 改善案B: ランダム選択モード
	if enable_random_selection:
		if len(available_files) <= window_size:
			updated_files = list(available_files)
			removed_files = []
		else:
			updated_files = _r.sample(available_files, window_size)
			removed_files = []
		if logger:
			logger.log_text(
				f"[buffer-window] random_selection selected={len(updated_files)} from {len(available_files)} available"
			)
		return updated_files, removed_files
	
	# 通常モード: 最新ファイル優先
	if not prev_files:
		# 初回起動: 利用可能なファイルから最新window_size件を選択
		try:
			files_with_mtime = [(f, os.path.getmtime(f)) for f in available_files]
			files_with_mtime.sort(key=lambda x: x[1])  # mtimeで昇順ソート
			sorted_files = [f for f, _ in files_with_mtime]
			if len(sorted_files) > window_size:
				updated_files = sorted_files[-window_size:]
				removed_files = []
			else:
				updated_files = sorted_files
				removed_files = []
		except Exception:
			# フォールバック: ファイル名ソート
			if len(available_files) > window_size:
				updated_files = available_files[-window_size:]
				removed_files = []
			else:
				updated_files = list(available_files)
				removed_files = []
		
		if logger:
			logger.log_text(
				f"[buffer-window] initial_load selected={len(updated_files)} from {len(available_files)} available window_size={window_size}"
			)
		return updated_files, removed_files
	
	# prev_filesから使用済みファイルを除外（改善案A）
	prev_set = set(prev_files)
	if used_files:
		used_set = set(used_files)
		prev_files = [f for f in prev_files if f not in used_set]
		prev_set = set(prev_files)
	
	# 利用可能なファイルから新しいファイルを検出
	new_files = [f for f in available_files if f not in prev_set]
	
	# 既存リストの末尾に新しいファイルを追加
	# 新しいファイルは更新日時でソート（古い順）
	try:
		new_files_with_mtime = [(f, os.path.getmtime(f)) for f in new_files]
		new_files_with_mtime.sort(key=lambda x: x[1])  # mtimeで昇順ソート
		new_files_sorted = [f for f, _ in new_files_with_mtime]
	except Exception:
		# フォールバック: ファイル名ソート
		new_files_sorted = sorted(new_files)
	
	# 既存ファイルと新規ファイルを結合
	# 既存ファイルはprev_filesの順序を維持、新規ファイルは時系列順に追加
	updated_files = list(prev_files) + new_files_sorted
	
	# ウィンドウサイズを超えたら先頭（古いファイル）から削除
	removed_files = []
	if len(updated_files) > window_size:
		removed_count = len(updated_files) - window_size
		removed_files = updated_files[:removed_count]
		updated_files = updated_files[-window_size:]
	
	if logger:
		# ログ出力: new_filesまたはremoved_filesがある場合
		if new_files or removed_files:
			logger.log_text(
				f"[buffer-window] prev={len(prev_files)} new={len(new_files)} "
				f"removed={len(removed_files)} current={len(updated_files)} window_size={window_size}"
			)
		# 削除されたファイルをログに出力（new_filesの有無に関係なく）
		if removed_files:
			try:
				for removed_file in removed_files:
					try:
						# ファイル名のみをログに出力（パスは長いため）
						removed_name = os.path.basename(removed_file)
						logger.log_text(f"[buffer-window-removed] {removed_name}")
					except Exception as e:
						# 個別ファイルのログ出力エラーは無視（次のファイルを続行）
						try:
							logger.log_text(f"[buffer-window-removed] ERROR: {str(e)}")
						except Exception:
							pass
			except Exception as e:
				# removed_filesのループ全体でエラーが発生した場合
				try:
					logger.log_text(f"[buffer-window-removed] ERROR in loop: {str(e)}")
				except Exception:
					pass
	
	return updated_files, removed_files


# --------------------------------------------------
# Active File Pool Utilities (modularized)
# --------------------------------------------------
def _scan_selfplay_files(data_dir: str):
	"""Return (all_files_sorted, pattern_attempts)
	pattern_attempts: List[Tuple[str, List[str]]] for logging counts.
	"""
	import os as _os
	# Use os.scandir for faster directory enumeration on large dirs
	pattern_attempts = [
		('selfplay_ep*_.joblib', []),
		('selfplay_ep*_mv*_.joblib', []),
		('selfplay_ep*.joblib', []),
	]
	try:
		with _os.scandir(data_dir) as it:
			for entry in it:
				if not entry.is_file():
					continue
				name = entry.name
				# skip in-progress temporary patterns
				if '.tmp.' in name:
					continue
				# primary pattern: ends with "_.joblib" (underscore before extension)
				if name.startswith('selfplay_ep') and name.endswith('_.joblib'):
					pattern_attempts[0][1].append(_os.path.join(data_dir, name))
				# secondary: contains _mv and endswith '_.joblib'
				elif name.startswith('selfplay_ep') and '_mv' in name and name.endswith('_.joblib'):
					pattern_attempts[1][1].append(_os.path.join(data_dir, name))
				# fallback: any selfplay_ep*.joblib
				elif name.startswith('selfplay_ep') and name.endswith('.joblib'):
					pattern_attempts[2][1].append(_os.path.join(data_dir, name))
	except Exception:
		# if scandir fails (e.g., dir missing), return empty lists as before
		pass

	# sort each collected list
	for i in range(len(pattern_attempts)):
		pattern_attempts[i] = (pattern_attempts[i][0], sorted(pattern_attempts[i][1]))
	# prefer first non-empty pattern in order
	all_files = pattern_attempts[0][1] or pattern_attempts[1][1] or pattern_attempts[2][1]
	return all_files, pattern_attempts


def _select_initial_active_files(all_files: List[str], cfg: Dict[str, Any], *, max_files: int | None, files_override: List[str] | None) -> List[str]:
	"""Encapsulate selection logic (legacy ingest vs active pool).
	
	改善版: newest_biasを使って最新ファイルと古いファイルからランダム選択を組み合わせる。
	- newest_bias=1.0: 全て最新ファイルから選択（従来動作）
	- newest_bias=0.5: 50%最新、50%古いファイルからランダム
	- newest_bias=0.0: 全てランダム選択
	"""
	import random as _r
	
	if files_override:
		return list(files_override)
	
	if not all_files:
		return []
	
	# 設定からパラメータ取得
	pool_size = int(cfg.get('active_file_pool_size', 0) or 0)
	newest_bias = float(cfg.get('active_file_newest_bias', 0.5) or 0.5)  # デフォルト50%最新
	
	# 使用するファイル数を決定
	try:
		eff_max_files = int(max_files) if (max_files is not None and int(max_files) > 0) else int(cfg.get('ingest_max_files') or 0)
	except Exception:
		eff_max_files = 0
	
	# pool_sizeが設定されていればそちらを優先、なければeff_max_filesを使用
	target_count = pool_size if pool_size > 0 else eff_max_files
	if target_count <= 0:
		target_count = len(all_files)
	target_count = min(target_count, len(all_files))
	
	# newest_biasに基づいて最新ファイルとランダム選択の割合を決定
	newest_count = int(target_count * newest_bias)
	random_count = target_count - newest_count
	
	# 最新ファイル（末尾から取得）
	latest_slice = all_files[-newest_count:] if newest_count > 0 else []
	
	# 残りは古いファイル（latest_sliceに含まれないもの）からランダム選択
	if random_count > 0:
		older_pool = all_files[:-newest_count] if newest_count > 0 else all_files
		if older_pool:
			random_picks = _r.sample(older_pool, min(random_count, len(older_pool)))
		else:
			random_picks = []
	else:
		random_picks = []
	
	# 結合して返す（最新ファイルを優先して末尾に配置）
	selected = random_picks + latest_slice
	return selected


def _load_samples_for_files(
	files: List[str], *,
	max_samples_per_file: int | None,
	seed: int | None,
	val_ratio: float,
	logger: TrainingLogger | None,
	cfg: Dict[str, Any],
	preloaded: Dict[str, Dict[str, List[Dict[str, Any]]]] | None = None,
	force_val_files: set | None = None,
):
	"""Load samples, shuffle, split, return (train_list, val_list, total_train, total_val).

	If ``preloaded`` is provided it should be a mapping from file path to the dict
	returned by ``_load_samples_from_file``; in that case the function will reuse
	those entries and avoid calling ``joblib.load`` again. When ``max_samples_per_file``
	is provided the reused parts will be truncated to that limit.
	"""
	import random as _r
	# Collect into per-file train/val lists based solely on file-level selection
	train_list: List[Dict[str, Any]] = []
	val_list: List[Dict[str, Any]] = []
	# Parallelize per-file loading where possible. If preloaded mapping is provided,
	# reuse entries; otherwise use joblib.Parallel to call _load_samples_from_file
	# concurrently. We measure per-file load time and emit lightweight events when
	# a logger is available.
	try:
		# Decide file-level val selection: use seed if provided for reproducibility
		# allow caller to force a specific set via `force_val_files`
		val_file_frac = float(cfg.get('active_file_val_fraction', 0.1) or 0.1)
		# ファイルが1つしかない場合、すべてのサンプルをtrainに含めるため、val_file_countを0にする
		if len(files) <= 1:
			val_file_count = 0
		else:
			val_file_count = max(1, int(len(files) * val_file_frac))
		if force_val_files is not None:
			val_files_set = set(force_val_files)
		else:
			# Use seed for reproducible file selection
			if seed is not None:
				_r_seed = _r.Random(seed)
				val_files_set = set(_r_seed.sample(files, val_file_count)) if val_file_count > 0 and len(files) >= val_file_count else set()
			else:
				val_files_set = set(_r.sample(files, val_file_count)) if val_file_count > 0 and len(files) >= val_file_count else set()
		if logger:
			try:
				logger.log_text(f"[file-split] selected {len(val_files_set)} files as val out of {len(files)}")
			except Exception:
				pass
		import time as _time, os as _os
		from joblib import Parallel, delayed
		# Profiling timestamps
		t_profile_start = _time.time()
		# number of workers configurable via cfg; default to (cpu_count - 1)
		_default_workers = 10
		try:
			_default_workers = max(1, ((_os.cpu_count() or 2) - 1))
		except Exception:
			_default_workers = 10
		n_jobs = int(cfg.get('resume_load_workers', _default_workers) or _default_workers)
		if n_jobs <= 0:
			try:
				import multiprocessing as _mp
				n_jobs = max(1, _mp.cpu_count() - 1)
			except Exception:
				n_jobs = 1

		# prepare list of files to actually load (deduplicate, skip preloaded)
		seen_load: set[str] = set()
		to_load: List[str] = []
		for fp in files:
			if preloaded is not None and fp in preloaded:
				continue
			if fp in seen_load:
				continue
			seen_load.add(fp)
			to_load.append(fp)

		def _load_with_timing(fp_local):
			"""Wrapper to load a file and return (fp, parts, load_time_s, size_bytes, samples_count)."""
			t0 = _time.time()
			parts_local = None
			try:
				parts_local = _load_samples_from_file(fp_local, max_samples=(max_samples_per_file if max_samples_per_file else None))
			except Exception:
				parts_local = {'train': [], 'val': []}
			t1 = _time.time()
			sz = 0
			try:
				sz = int(_os.path.getsize(fp_local))
			except Exception:
				sz = 0
			tr = parts_local.get('train', []) or []
			vl = parts_local.get('val', []) or []
			count = len(tr) + len(vl)
			return (fp_local, parts_local, float(t1 - t0), int(sz), int(count))

		results = []
		if to_load:
			# Use loky backend for true parallelization (avoids GIL limitations of threading)
			# This enables parallel decompression of joblib files across multiple CPU cores
			try:
				workers = max(1, min(n_jobs, len(to_load)))
				results = Parallel(n_jobs=workers, backend='loky')(delayed(_load_with_timing)(fp) for fp in to_load)
			except Exception:
				# fallback serial
				results = [_load_with_timing(fp) for fp in to_load]
		t_load_done = _time.time()
		if logger:
			try:
				logger.log_text(f"[profile-load] parallel_load took={t_load_done - t_profile_start:.2f}s files={len(to_load)}")
			except Exception:
				pass
		# build mapping fp -> parts
		parts_map: Dict[str, tuple] = {}
		for fp, parts_local, load_t, sz, cnt in results:
			if fp in parts_map:
				# 既に登録済み（重複読み込み）は無視
				continue
			parts_map[fp] = (parts_local, load_t, sz, cnt)
		# include preloaded entries with zero load time
		if preloaded is not None:
			for fp in files:
				if fp in preloaded and fp not in parts_map:
					parts_map[fp] = (preloaded[fp], 0.0, 0, (len(preloaded[fp].get('train', []) or []) + len(preloaded[fp].get('val', []) or [])))

		# iterate in original order and append samples to train_list or val_list
		# Note: max_samples_per_file truncation is already applied in _load_samples_from_file
		for fp in files:
			entry = parts_map.get(fp, ({'train': [], 'val': []}, 0.0, 0, 0))
			parts = entry[0]
			try:
				tr = parts.get('train', []) or []
				vl = parts.get('val', []) or []
			except Exception:
				tr = []
				vl = []
			# Apply max_samples_per_file limit to preloaded data as well
			if max_samples_per_file is not None and max_samples_per_file > 0:
				total_samples = len(tr) + len(vl)
				if total_samples > max_samples_per_file:
					combined = tr + vl
					combined = _r.sample(combined, max_samples_per_file)
					# Re-split into train/val (keep original ratio if possible)
					if len(tr) > 0 and len(vl) > 0:
						orig_val_ratio = len(vl) / total_samples
						new_val_count = int(max_samples_per_file * orig_val_ratio)
						vl = combined[:new_val_count]
						tr = combined[new_val_count:]
					else:
						# All train or all val
						tr = combined if len(tr) > 0 else []
						vl = combined if len(vl) > 0 else []
			if fp in val_files_set:
				try:
					val_list.extend(tr)
					val_list.extend(vl)
				except Exception:
					pass
			else:
				try:
					train_list.extend(tr)
				except Exception:
					pass
		# メモリ効率化: parts_mapをクリア（train_list/val_listに追加済みなので不要）
		try:
			del parts_map
		except Exception:
			pass
		t_split_done = _time.time()
		if logger:
			try:
				logger.log_text(f"[profile-load] list_building took={t_split_done - t_load_done:.2f}s train={len(train_list)} val={len(val_list)}")
			except Exception:
				pass
	except Exception:
		# If anything went wrong with parallel path, fallback to serial per-file load
		train_list = []
		val_list = []
		for fp in files:
			parts = None
			try:
				if preloaded is not None and fp in preloaded:
					parts = preloaded[fp]
				else:
					parts = _load_samples_from_file(fp, max_samples=(max_samples_per_file if max_samples_per_file else None))
			except Exception:
				parts = {'train': [], 'val': []}
			try:
				tr = parts.get('train', []) or []
				vl = parts.get('val', []) or []
			except Exception:
				tr = []
				vl = []
			if fp in val_files_set:
				try:
					val_list.extend(tr)
					val_list.extend(vl)
				except Exception:
					pass
			else:
				try:
					train_list.extend(tr)
				except Exception:
					pass

	# 毎セッションで異なるシャッフルを実現するため、時刻ベースの動的seedを使用
	# シャッフルは train_list のみ行い、val_list はファイル単位で保持する
	try:
		import time as _time
		base_seed = int(seed) if seed is not None else 0
		dynamic_seed = (base_seed + int(_time.time() * 1000)) % (2**31)
		t_shuffle_start = _time.time()
		# Use numpy.random.permutation for faster shuffling than random.shuffle
		try:
			import numpy as _np
			rng = _np.random.default_rng(dynamic_seed)
			indices = rng.permutation(len(train_list))
			train_list = [train_list[i] for i in indices]
		except Exception:
			# Fallback to Python random.shuffle
			_r.Random(dynamic_seed).shuffle(train_list)
		t_shuffle_done = _time.time()
		if logger:
			try:
				logger.log_text(f"[shuffle] dynamic_seed={dynamic_seed} train_samples={len(train_list)} val_samples={len(val_list)} took={t_shuffle_done - t_shuffle_start:.2f}s")
			except Exception:
				pass
	except Exception:
		try:
			_r.shuffle(train_list)
		except Exception:
			pass

	return train_list, val_list, len(train_list), len(val_list), val_files_set


def _rebuild_replay(
	shared_rb: ReplayBuffer,
	train_part: List[Dict[str, Any]],
	val_part: List[Dict[str, Any]],
	logger=None,
	*,
	chunk_size: int | None = None,  # legacy arg (ignored; no chunking)
):
	"""Clear and repopulate replay buffer with provided parts."""
	import time as _time
	t0 = _time.time()
	try:
		shared_rb.clear()
	except Exception:
		pass
	t1 = _time.time()
	train_total = 0
	val_total = 0
	t2 = _time.time()
	# デバッグ: train_partとval_partのplayer_id=0のサンプル数を確認
	try:
		train_pid0_count = sum(1 for s in train_part if isinstance(s, dict) and s.get('player_id') == 0)
		val_pid0_count = sum(1 for s in val_part if isinstance(s, dict) and s.get('player_id') == 0)
		print(f"[DEBUG_VALUE_MIX] _rebuild_replay: train_part={len(train_part)} (pid=0: {train_pid0_count}), val_part={len(val_part)} (pid=0: {val_pid0_count})")
	except Exception:
		pass
	# 全件を一括で処理する（チャンク分割は廃止）
	# メモリ消費を抑えたい場合は上流でリストを絞り込むこと

	t3 = _time.time()
	# Use ReplayBuffer.extend when available
	try:
		uids = []
		t4 = _time.time()
		if logger:
			logger.log_text(f"[rebuild] Starting extend: train={len(train_part)} val={len(val_part)}")
		if hasattr(shared_rb, 'extend'):
			# splitタグを事前設定（高速化）
			for s in train_part:
				if isinstance(s, dict):
					s['split'] = 'train'
			for s in val_part:
				if isinstance(s, dict):
					s['split'] = 'val'
			
			# チャンク分割して進捗表示（大量サンプル時の体感速度改善）
			# 重要: valサンプルを先に追加してバッファから押し出されないようにする
			chunk_size = 50000
			if val_part:
				if logger:
					logger.log_text(f"[rebuild] Extending val samples first: {len(val_part)}")
				for i in range(0, len(val_part), chunk_size):
					chunk = val_part[i:i+chunk_size]
					val_uids = shared_rb.extend(chunk)
					if isinstance(val_uids, list):
						uids.extend(val_uids)
					val_total += len(chunk)
			
			if train_part:
				if logger:
					logger.log_text(f"[rebuild] Extending train samples: {len(train_part)}")
				for i in range(0, len(train_part), chunk_size):
					chunk = train_part[i:i+chunk_size]
					train_uids = shared_rb.extend(chunk)
					if isinstance(train_uids, list):
						uids.extend(train_uids)
					train_total += len(chunk)
			t5 = _time.time()
			# Debug timing (always log for now to diagnose)
			if logger:
				try:
					logger.log_text(
						f"[rebuild-timing] clear={t1-t0:.2f}s set_split={t3-t2:.2f}s extend={t5-t4:.2f}s "
						f"total={t5-t0:.2f}s samples={train_total+val_total} chunk_size=-1"
					)
				except Exception:
					pass
		else:
			t5a = _time.time()
			for s in train_part:
				if isinstance(s, dict):
					try:
						s['split'] = 'train'
					except Exception:
						pass
				try:
					shared_rb.append(s)
					train_total += 1
				except Exception:
					pass
			for s in val_part:
				if isinstance(s, dict):
					try:
						s['split'] = 'val'
					except Exception:
						pass
				try:
					shared_rb.append(s)
					val_total += 1
				except Exception:
					pass
			t6a = _time.time()
			# Debug timing
			if logger:
				try:
					logger.log_text(
						f"[rebuild-timing] clear={t1-t0:.2f}s set_split={t3-t2:.2f}s append_loop={t6a-t5a:.2f}s "
						f"total={t6a-t0:.2f}s samples={train_total+val_total} chunk_size=-1"
					)
				except Exception:
					pass
	except Exception:
		# fallback to per-item append in case of unexpected errors
		train_total = 0
		val_total = 0
		t7 = _time.time()
		for s in train_part:
			if isinstance(s, dict):
				try:
					s['split'] = 'train'
				except Exception:
					pass
			try:
				shared_rb.append(s)
				train_total += 1
			except Exception:
				pass
		for s in val_part:
			if isinstance(s, dict):
				try:
					s['split'] = 'val'
				except Exception:
					pass
			try:
				shared_rb.append(s)
				val_total += 1
			except Exception:
				pass
		t8 = _time.time()
		# Debug timing
		if logger:
			try:
				logger.log_text(
					f"[rebuild-timing] clear={t1-t0:.2f}s set_split={t3-t2:.2f}s fallback_append={t8-t7:.2f}s "
					f"total={t8-t0:.2f}s samples={train_total+val_total} chunk_size=-1"
				)
			except Exception:
				pass
	return train_total, val_total


def _compute_dynamic_samples_per_file(cfg: Dict[str, Any], train_updates_cum: int, num_files: int, logger: TrainingLogger | None = None) -> int | None:
	"""Compute dynamic samples per file based on total_updates^exponent.
	
	Deprecated: Use _compute_dynamic_file_and_sample_params instead for more accurate control.
	"""
	try:
		if not bool(cfg.get('enable_dynamic_sample_sizing', False)):
			return None
		
		exponent = float(cfg.get('dynamic_sample_exponent', 0.75))
		base = float(cfg.get('dynamic_sample_base', 1.0))
		min_per_file = int(cfg.get('dynamic_sample_min_per_file', 50))
		max_per_file = int(cfg.get('dynamic_sample_max_per_file', 10000))
		
		if train_updates_cum <= 0 or num_files <= 0:
			return None
		
		target_total_samples = base * (train_updates_cum ** exponent)
		samples_per_file = int(target_total_samples / num_files)
		samples_per_file = max(min_per_file, min(samples_per_file, max_per_file))
		
		if logger:
			logger.log_text(f"[dynamic-sizing] updates={train_updates_cum} target_samples={target_total_samples:.0f} files={num_files} samples_per_file={samples_per_file}")
		
		return samples_per_file
	except Exception as e:
		if logger:
			logger.log_text(f"[WARN] dynamic sample sizing computation failed: {e}")
		return None


def _compute_dynamic_file_and_sample_params(cfg: Dict[str, Any], train_updates_cum: int, available_files_count: int, logger: TrainingLogger | None = None) -> tuple[int, int] | None:
	"""Compute both file count and samples per file dynamically.
	
	Returns:
		(num_files, samples_per_file) or None if dynamic sizing is disabled
	"""
	try:
		# 固定サンプル数が設定されている場合は優先
		fixed_total = int(cfg.get('fixed_total_samples', 0) or 0)
		if fixed_total > 0:
			min_per_file = int(cfg.get('dynamic_sample_min_per_file', 200))
			max_per_file = int(cfg.get('dynamic_sample_max_per_file', 2500))
			samples_per_file_target = int(cfg.get('dynamic_samples_per_file_target', 2000))
			min_files = int(cfg.get('dynamic_min_files', 5))
			max_files = int(cfg.get('dynamic_max_files', 200))
			
			# 固定総サンプル数から必要なファイル数を計算
			num_files = int(fixed_total / samples_per_file_target)
			num_files = max(min_files, min(num_files, max_files, available_files_count))
			
			# ファイル数に基づいてファイルあたりのサンプル数を再計算
			samples_per_file = int(fixed_total / num_files) if num_files > 0 else min_per_file
			samples_per_file = max(min_per_file, min(samples_per_file, max_per_file))
			
			if logger:
				logger.log_text(f"[fixed-sizing] fixed_total_samples={fixed_total} files={num_files} samples_per_file={samples_per_file}")
			
			return (num_files, samples_per_file)
		
		if not bool(cfg.get('enable_dynamic_sample_sizing', False)):
			return None
		
		exponent = float(cfg.get('dynamic_sample_exponent', 0.75))
		base = float(cfg.get('dynamic_sample_base', 1.0))
		min_per_file = int(cfg.get('dynamic_sample_min_per_file', 200))
		max_per_file = int(cfg.get('dynamic_sample_max_per_file', 2500))
		
		if train_updates_cum <= 0:
			return None
		
		# 目標総サンプル数を計算
		target_total_samples = base * (train_updates_cum ** exponent)
		
		# ファイル数も動的に調整するか
		if bool(cfg.get('enable_dynamic_file_count', True)):
			samples_per_file_target = int(cfg.get('dynamic_samples_per_file_target', 2000))
			min_files = int(cfg.get('dynamic_min_files', 5))
			max_files = int(cfg.get('dynamic_max_files', 50))
			
			# 必要なファイル数を計算
			num_files = int(target_total_samples / samples_per_file_target)
			num_files = max(min_files, min(num_files, max_files, available_files_count))
			
			# ファイル数に基づいてファイルあたりのサンプル数を再計算
			samples_per_file = int(target_total_samples / num_files)
			samples_per_file = max(min_per_file, min(samples_per_file, max_per_file))
		else:
			# ファイル数固定モード（従来の動作）
			num_files = min(int(cfg.get('ingest_max_files', 40)), available_files_count)
			samples_per_file = int(target_total_samples / num_files) if num_files > 0 else min_per_file
			samples_per_file = max(min_per_file, min(samples_per_file, max_per_file))
		
		if logger:
			logger.log_text(f"[dynamic-sizing] updates={train_updates_cum} target_samples={target_total_samples:.0f} files={num_files} samples_per_file={samples_per_file}")
		
		return (num_files, samples_per_file)
	except Exception as e:
		if logger:
			logger.log_text(f"[WARN] dynamic file/sample computation failed: {e}")
		return None


def _refresh_active_pool(active_files: List[str], all_files: List[str], cfg: Dict[str, Any], shared_rb: ReplayBuffer, val_ratio: float, seed: int | None, max_samples_per_file: int | None, logger: TrainingLogger | None, train_updates_cum: int = 0, data_dir: str | None = None):
	"""Active file pool refresh.

	rebuild モード: 全ファイル再サンプリングしバッファをクリアして再構築。
	incremental モード: 退避ファイルのサンプルは残し、新規追加ファイルのみ読み込み append。
	ReplayBuffer.maxlen による自然な古いサンプルのドロップを利用してリングバッファ的挙動を得る。
	"""
	import random as _r
	incremental = bool(cfg.get('active_file_incremental_refresh', True))
	pool_size = int(cfg.get('active_file_pool_size', 0) or 0)
	newest_bias = float(cfg.get('active_file_newest_bias', 0.0) or 0.0)
	refresh_fraction = float(cfg.get('active_file_pool_refresh_fraction', 0.0) or 0.0)
	current_set = set(active_files)
	available = [f for f in all_files if f not in current_set]
	replace_n = max(1, int(len(active_files) * refresh_fraction)) if (refresh_fraction > 0.0 and active_files) else (max(1, int(len(active_files) * 0.3)) if active_files else 0)
	replace_n = min(replace_n, len(active_files)) if active_files else 0
	evict = _r.sample(active_files, replace_n) if replace_n > 0 else []
	remain = [f for f in active_files if f not in evict]
	add_n = min(replace_n, len(available))
	new_add = _r.sample(available, add_n) if add_n > 0 else []
	if new_add:
		seen_active = set(remain)
		seen_new: set[str] = set()
		filtered_add: List[str] = []
		for fp in new_add:
			if fp in seen_active or fp in seen_new:
				continue
			seen_new.add(fp)
			filtered_add.append(fp)
		new_add = filtered_add
	if newest_bias > 0.0 and pool_size > 0:
		bias_count = int(min(pool_size, max(0, int(pool_size * newest_bias))))
		latest_slice = all_files[-bias_count:] if bias_count > 0 else []
		remain = [f for f in remain if f not in latest_slice]
		new_add = [f for f in new_add if f not in latest_slice]
		combined = latest_slice + remain + new_add
		active_files_new = combined[:pool_size]
	else:
		active_files_new = remain + new_add
	
	# 動的サンプルサイジングを適用
	dynamic_result = _compute_dynamic_file_and_sample_params(cfg, train_updates_cum, len(all_files), logger)
	if dynamic_result is not None:
		desired_files, dynamic_samples = dynamic_result
		# プールサイズを動的に調整（既存ファイルを維持しつつ目標に近づける）
		if len(active_files_new) != desired_files and len(all_files) >= desired_files:
			if len(active_files_new) < desired_files:
				# ファイルを追加
				add_more = desired_files - len(active_files_new)
				available_more = [f for f in all_files if f not in active_files_new]
				if available_more:
					import random as _r2
					extra = _r2.sample(available_more, min(add_more, len(available_more)))
					active_files_new = active_files_new + extra
			elif len(active_files_new) > desired_files:
				# ファイルを削減（最も古いものから）
				active_files_new = active_files_new[-desired_files:]
		max_samples_per_file = dynamic_samples
	
	if not incremental:
		train_part, val_part, tcount, vcount, val_files_set = _load_samples_for_files(active_files_new, max_samples_per_file=max_samples_per_file, seed=seed, val_ratio=val_ratio, logger=logger, cfg=cfg)
		chunk_sz = int(cfg.get('rebuild_chunk_size', 0) or 0)
		loaded_train, loaded_val = _rebuild_replay(shared_rb, train_part, val_part, logger, chunk_size=chunk_sz)
		# メモリ効率化: train_partとval_partを明示的にクリア（ReplayBufferに追加済みなので不要）
		try:
			del train_part
			del val_part
		except Exception:
			pass

		# If data_dir provided, persist removal of val files so they won't be reused
		if data_dir and val_files_set:
			try:
				# remove selected val files from active pool and mark them as used
				new_active = [f for f in active_files_new if f not in val_files_set]
				_save_buffer_state(data_dir, new_active, used_files=list(val_files_set), logger=logger)
				active_files_new = new_active
			except Exception:
				try:
					if logger:
						logger.log_text('[WARN] failed to persist val file removal to buffer_state')
				except Exception:
					pass

		if logger:
			logger.log_text(f"[active-pool] refresh(rebuild) replaced={replace_n} added={add_n} pool_size={len(active_files_new)} replay_size={len(shared_rb)} train={loaded_train} val={loaded_val}")
		return active_files_new, {'replaced': replace_n, 'added': add_n, 'train': loaded_train, 'val': loaded_val, 'mode': 'rebuild'}
	# incremental モード: 新規追加ファイルのみロードして追加
	app_train = 0
	app_val = 0
	try:
		# まず各ファイルからサンプルを収集（再読み込みを避ける）
		per_file = []  # List[Tuple[List[dict], List[dict]]]
		total_new = 0
		seen_new_add: set[str] = set()
		ordered_new: List[str] = []
		for fp in new_add:
			if fp in seen_new_add:
				continue
			seen_new_add.add(fp)
			ordered_new.append(fp)
		parts_map: Dict[str, tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
		if ordered_new:
			from joblib import Parallel, delayed
			import time as _time
			# Align worker count with resume_load_workers to keep behaviour consistent
			_default_workers = 10
			try:
				_default_workers = max(1, ((os.cpu_count() or 2) - 1))
			except Exception:
				_default_workers = 10
			n_jobs = int(cfg.get('resume_load_workers', _default_workers) or _default_workers)
			if n_jobs <= 0:
				try:
					import multiprocessing as _mp
					n_jobs = max(1, _mp.cpu_count() - 1)
				except Exception:
					n_jobs = 1
			max_samples_arg = int(max_samples_per_file) if (max_samples_per_file is not None and max_samples_per_file > 0) else None
			def _load_incremental(fp_local: str):
				t_start = _time.time()
				parts_local = {'train': [], 'val': []}
				try:
					parts_local = _load_samples_from_file(fp_local, max_samples=max_samples_arg)
				except Exception:
					parts_local = {'train': [], 'val': []}
				t_end = _time.time()
				tr_local = parts_local.get('train', []) or []
				vl_local = parts_local.get('val', []) or []
				return fp_local, tr_local, vl_local, float(t_end - t_start)
			t_parallel_start = _time.time()
			try:
				workers = max(1, min(n_jobs, len(ordered_new)))
				results = Parallel(n_jobs=workers, backend='threading')(delayed(_load_incremental)(fp) for fp in ordered_new)
			except Exception:
				results = [_load_incremental(fp) for fp in ordered_new]
			t_parallel_end = _time.time()
			for fp_local, tr_local, vl_local, _dur in results:
				if fp_local not in parts_map:
					parts_map[fp_local] = (tr_local, vl_local)
			if logger:
				try:
					logger.log_text(f"[profile-load] incremental_parallel took={t_parallel_end - t_parallel_start:.2f}s files={len(ordered_new)}")
				except Exception:
					pass
		for fp in ordered_new:
			tr, vl = parts_map.get(fp, ([], []))
			per_file.append((tr, vl))
			total_new += len(tr) + len(vl)
		# バッファ容量に応じて事前選別
		cap = int(getattr(shared_rb, 'maxlen', 0) or 0)
		cur = int(len(shared_rb))
		selected = None
		if cap > 0 and total_new >= cap:
			# ケース1: 追加分だけで満杯を超える → 全クリアして最新 cap 件のみ
			try:
				shared_rb.clear()
			except Exception:
				pass
			from collections import deque as _dq
			ring = _dq(maxlen=cap)
			for tr, vl in per_file:
				for s in tr:
					if isinstance(s, dict):
						s['split'] = 'train'
						ring.append(s)
				for s in vl:
					if isinstance(s, dict):
						s['split'] = 'val'
						ring.append(s)
			selected = list(ring)
		elif cap > 0 and (cur + total_new > cap):
			# ケース2: 溢れる分だけ先に古いデータをまとめて捨てる
			try:
				need_keep = cap - total_new
				if need_keep < 0:
					need_keep = 0
				if hasattr(shared_rb, 'shrink_to_size'):
					_ = shared_rb.shrink_to_size(need_keep)
			except Exception:
				pass
			# そのまま全件追加対象
			selected = []
			for tr, vl in per_file:
				for s in tr:
					if isinstance(s, dict):
						s['split'] = 'train'
						selected.append(s)
				for s in vl:
					if isinstance(s, dict):
						s['split'] = 'val'
						selected.append(s)
		else:
			# ケース3: 余裕がある → 全件追加対象
			selected = []
			for tr, vl in per_file:
				for s in tr:
					if isinstance(s, dict):
						s['split'] = 'train'
						selected.append(s)
				for s in vl:
					if isinstance(s, dict):
						s['split'] = 'val'
						selected.append(s)
		# 一括追加（append ループ）
		# Use bulk extend if available to reduce per-item overhead
		try:
			if hasattr(shared_rb, 'extend'):
				shared_rb.extend(selected)
				app_train = sum(1 for s in selected if s.get('split') != 'val')
				app_val = sum(1 for s in selected if s.get('split') == 'val')
			else:
				for s in selected:
					try:
						shared_rb.append(s)
						if s.get('split') == 'val':
							app_val += 1
						else:
							app_train += 1
					except Exception:
						pass
		except Exception:
			# fallback to per-item append
			for s in selected:
				try:
					shared_rb.append(s)
					if s.get('split') == 'val':
						app_val += 1
					else:
						app_train += 1
				except Exception:
					pass
		# メモリ効率化: selectedリストとper_fileリストを明示的にクリア（ReplayBufferに追加済みなので不要）
		try:
			del selected
		except Exception:
			pass
		try:
			del per_file
		except Exception:
			pass
	except Exception as e:
		if logger:
			logger.log_text(f"[WARN] incremental refresh load failed: {e}")
	if logger:
		logger.log_text(f"[active-pool] refresh(incremental) replaced={replace_n} added={add_n} appended_train={app_train} appended_val={app_val} pool_size={len(active_files_new)} replay_size={len(shared_rb)}")
	return active_files_new, {'replaced': replace_n, 'added': add_n, 'train': app_train, 'val': app_val, 'mode': 'incremental'}


def _load_config(base: Dict[str, Any], path: str | None) -> Dict[str, Any]:
	cfg = dict(base)
	if path and os.path.isfile(path):
		try:
			with open(path, 'r', encoding='utf-8') as f:
				user = json.load(f)
			if isinstance(user, dict):
				cfg.update(user)
		except Exception as e:
			print(f"[WARN] config json load failed: {e}")
	return cfg


def _resolve_device(dev: str | None) -> str:
	try:
		import torch  # type: ignore
	except Exception:
		return 'cpu'
	if not dev or dev == 'auto':
		return 'cuda' if (torch.cuda.is_available() and getattr(torch.version, 'cuda', None)) else 'cpu'
	# 明示的に cuda* 要求されたが CPU ビルド or 非利用の場合はフォールバック
	if isinstance(dev, str) and dev.startswith('cuda'):
		if (not torch.cuda.is_available()) or (getattr(torch.version, 'cuda', None) is None):
			print('[WARN] requested CUDA but torch not compiled with CUDA -> fallback to cpu')
			return 'cpu'
	return dev


def _atomic_write_json(path: str, obj: Dict[str, Any]):
	tmp = path + '.tmp'
	try:
		with open(tmp, 'w', encoding='utf-8') as f:
			json.dump(obj, f, ensure_ascii=False, indent=2)
		os.replace(tmp, path)
	except Exception:
		try:
			if os.path.exists(tmp):
				os.remove(tmp)
		except Exception:
			pass
		with open(path, 'w', encoding='utf-8') as f:
			json.dump(obj, f, ensure_ascii=False, indent=2)


def _list_selfplay_files(data_dir: str) -> List[str]:
	files: List[str] = []
	try:
		for name in os.listdir(data_dir):
			if name.startswith('selfplay_ep') and name.endswith('.joblib'):
				files.append(os.path.join(data_dir, name))
	except Exception:
		pass
	files.sort()  # 時系列昇順
	return files


def _load_samples_from_file(path: str, *, max_samples: int | None = None) -> Dict[str, List[Dict[str, Any]]]:
	try:
		# Prefer standard load first — experimentally this is much faster
		# for compressed joblib dumps on many environments. Only if that fails
		# attempt a mmap-mode load as a fallback.
		try:
			payload = joblib.load(path)
		except Exception:
			try:
				with warnings.catch_warnings():
					warnings.filterwarnings(
						'ignore',
						message=r'.*mmap_mode\s*"r"\s*is not compatible with compressed file.*',
						category=UserWarning,
					)
					payload = joblib.load(path, mmap_mode='r')
			except Exception as e:
				print(f"[WARN] failed to load {path}: {e}")
				return {'train': [], 'val': []}
	except Exception as e:
		print(f"[WARN] failed to load {path}: {e}")
		return {'train': [], 'val': []}
	samples = payload.get('samples') or []
	train_list: List[Dict[str, Any]] = []
	val_list: List[Dict[str, Any]] = []
	# Treat all samples as train by default; file-level selection will move
	# entire files to validation when needed.
	for s in samples:
		if not isinstance(s, dict):
			continue
		try:
			s['source_file'] = path
		except Exception:
			pass
		train_list.append(s)
	if max_samples is not None and max_samples > 0:
		if len(train_list) > max_samples:
			train_list = random.sample(train_list, max_samples)
	return {'train': train_list, 'val': val_list}


def _load_with_timing_worker(fp_local: str, max_samples_per_file: int | None):
	"""Module-level worker for Parallel to avoid pickling nested functions.

	Returns: (fp, parts, load_time_s, size_bytes, samples_count)
	"""
	import time as _time, os as _os
	t0 = _time.time()
	try:
		parts_local = _load_samples_from_file(
			fp_local, max_samples=(max_samples_per_file if max_samples_per_file else None)
		)
	except Exception:
		parts_local = {'train': [], 'val': []}
	t1 = _time.time()
	sz = 0
	try:
		sz = int(_os.path.getsize(fp_local))
	except Exception:
		sz = 0
	tr = parts_local.get('train', []) or []
	vl = parts_local.get('val', []) or []
	count = len(tr) + len(vl)
	return (fp_local, parts_local, float(t1 - t0), int(sz), int(count))


def _load_meta(data_dir: str) -> Dict[str, Any]:
	meta_path = os.path.join(data_dir, 'meta.json')
	if os.path.isfile(meta_path):
		try:
			with open(meta_path, 'r', encoding='utf-8') as f:
				return json.load(f) or {}
		except Exception:
			return {}
	return {}


def _update_meta(data_dir: str, meta: Dict[str, Any]):
	path = os.path.join(data_dir, 'meta.json')
	# Try to use async IO writer if enabled in config
	try:
		from utils.async_io import get_async_io
		# Attempt to honor config by passing None (caller may pass cfg when available)
		async_io = get_async_io(None)
		if async_io is not None:
			try:
				txt = json.dumps(meta, ensure_ascii=False, indent=2)
				async_io.enqueue_text_write(txt, path)
				return
			except Exception:
				pass
	except Exception:
		pass
	# fallback synchronous atomic write
	_atomic_write_json(path, meta)


def _run_validation(learner: AlphaZeroAgent, logger: TrainingLogger, cfg: Dict[str, Any], *, batch_size: int | None = None, note: str | None = None):
	"""Validate on current 'val' split and log results to CSV via TrainingLogger.

	- Uses cfg['val_batch_size'] or cfg['batch_size'] as fallback.
	- Ensures val_* columns are present by invoking logger.log_validation prior to train rows.
	"""
	try:
		bs = int(batch_size or cfg.get('val_batch_size') or cfg.get('batch_size', 256) or 256)
	except Exception:
		bs = 256
	# Ensure learner.config respects trainer-level validation overrides
	try:
		if hasattr(learner, 'config') and isinstance(learner.config, dict):
			for _k in ('val_max_samples', 'val_use_full_split', 'val_batch_size'):
				if _k in cfg:
					try:
						learner.config[_k] = cfg[_k]
					except Exception:
						pass
	except Exception:
		pass
	try:
		vinfo = learner.validate_step(batch_size=bs)
	except Exception:
		vinfo = {"policy_loss": None, "value_loss": None, "hand_pred_loss": None}
	# Always pass the validation info to the logger so that
	# the validation columns are present in `train_updates.csv` even when
	# the validation pool is empty. validate_step may return a dict with
	# None values when no validation data is available; logger.log_validation
	# will record those None values (and write a placeholder row if needed).
	try:
		if isinstance(vinfo, dict):
			logger.log_validation(vinfo)
			# If all primary metrics are None, decide whether to warn.
			no_val = not (vinfo.get("policy_loss") is not None or vinfo.get("value_loss") is not None or vinfo.get("hand_pred_loss") is not None)
			# If the config explicitly sets val_split_ratio == 0, the absence of validation
			# data is intentional -> do not spam warnings. Otherwise, warn once per logger instance.
			try:
				val_ratio_cfg = float(cfg.get('val_split_ratio', 0.0) or 0.0)
			except Exception:
				val_ratio_cfg = 0.0
			if no_val:
				if val_ratio_cfg <= 0.0:
					# intentional: skip warning
					try:
						# still ensure warning flag is cleared so future real warnings can appear
						if getattr(logger, '_val_warning_logged', False):
							setattr(logger, '_val_warning_logged', False)
					except Exception:
						pass
				else:
					if not getattr(logger, '_val_warning_logged', False):
						try:
							logger.log_text("[val] no validation metrics (val split may be empty)")
						except Exception:
							pass
						try:
							setattr(logger, '_val_warning_logged', True)
						except Exception:
							pass
			else:
				try:
					if getattr(logger, '_val_warning_logged', False):
						setattr(logger, '_val_warning_logged', False)
				except Exception:
					pass
		else:
			# non-dict return: still record empty metrics and warn once
			try:
				if not getattr(logger, '_val_warning_logged', False):
					logger.log_text("[val] validate_step returned non-dict result; skipping detailed val logging")
					setattr(logger, '_val_warning_logged', True)
			except Exception:
				pass
			logger.log_validation({"policy_loss": None, "value_loss": None, "hand_pred_loss": None})
	except Exception:
		try:
			logger.log_text("[val] validation logging failed")
		except Exception:
			pass
	return vinfo
		


def _save_checkpoint(bundle, cfg: Dict[str, Any], model_version: int, version_tag: str | None = None, logger=None):
	os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
	model = bundle.model
	learner = bundle.agents[cfg.get('learning_player_id', 0)]
	latest_path = cfg.get('checkpoint_path', os.path.join(cfg['checkpoint_dir'], 'policy_value_latest.pt'))
	def _atomic(pt: str):
		tmp = pt + '.tmp'
		try:
			model.save(tmp, force_sync=True, logger=logger)  # type: ignore[arg-type]
			os.replace(tmp, pt)
		except Exception:
			try:
				if os.path.exists(tmp):
					os.remove(tmp)
			except Exception:
				pass
			model.save(pt, logger=logger)  # type: ignore[arg-type]
	if model is not None:
		_atomic(latest_path)
	if version_tag:
		tag_path = os.path.join(cfg['checkpoint_dir'], f'policy_value_{version_tag}.pt')
		if model is not None:
			_atomic(tag_path)
	# optimizer & scheduler
	try:
		opt_path = os.path.join(cfg['checkpoint_dir'], 'optimizer_latest.pt')
		if hasattr(learner, 'ensure_optimizer'):
			learner.ensure_optimizer()
		if hasattr(learner, 'save_optimizer'):
			learner.save_optimizer(opt_path)
	except Exception as e:
		print(f"[WARN] optimizer save failed: {e}")
	try:
		sch_path = os.path.join(cfg['checkpoint_dir'], 'scheduler_latest.pt')
		if hasattr(learner, 'ensure_scheduler'):
			learner.ensure_scheduler()
		if hasattr(learner, 'save_scheduler'):
			learner.save_scheduler(sch_path)
	except Exception as e:
		print(f"[WARN] scheduler save failed: {e}")
	# metadata.json (モデル構成ハッシュ)
	try:
		meta = {
			'model_version': model_version,
			'saved_at': time.time(),
			'use_full_features': bool(cfg.get('use_full_features')),
			'num_players': cfg.get('num_players'),
			'hidden_size': cfg.get('hidden_size'),
			'max_policy_size': cfg.get('max_policy_size'),
			'seed': cfg.get('seed'),
			'config_md5': hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
		}
		mpath = os.path.join(cfg['checkpoint_dir'], 'metadata.json')
		_atomic_write_json(mpath, meta)
	except Exception as e:
		print(f"[WARN] metadata save failed: {e}")


def train_loop(cfg: Dict[str, Any], *, data_dir: str, log_dir: str, max_files: int | None, max_samples_per_file: int | None, updates: int, version_interval: int, files_override: List[str] | None = None):
	# Early return if updates=0 to skip heavy initialization (model loading, sample loading, etc.)
	updates = int(updates)
	
	# updates <= 0 でも buffer_state モードの場合、active_files を更新して保存する
	buffer_window_size = int(cfg.get('buffer_window_size', 0) or 0)
	if updates <= 0 and buffer_window_size > 0 and not files_override:
		try:
			# buffer_state の更新のみ実行（軽量処理）
			all_files, _ = _scan_selfplay_files(data_dir)
			buffer_state = _load_buffer_state(data_dir)
			prev_files = buffer_state.get('active_files', [])
			used_files = buffer_state.get('used_files', [])
			enable_random_selection = bool(cfg.get('buffer_window_random_selection', True))
			active_files, _ = _update_buffer_window(
				prev_files, all_files, buffer_window_size,
				used_files=used_files,
				enable_random_selection=enable_random_selection,
				logger=None  # logger は初期化前なので None
			)
			# active_files のみ更新（used_files は更新しない、学習が実行されていないため）
			_save_buffer_state(data_dir, active_files, used_files=None, logger=None)
		except Exception as e:
			# buffer_state 更新エラーは無視して続行
			print(f"[WARN] buffer_state update failed (updates=0): {e}")
	
	if updates <= 0:
		# Load meta to get cumulative updates for logging
		meta = _load_meta(data_dir)
		train_updates_cum = int(meta.get('train_updates_cumulative', 0) or 0)
		print(f"[INFO] training finished updates={updates} total_updates={train_updates_cum}")
		return
	
	os.makedirs(data_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(cfg['checkpoint_dir'], exist_ok=True)

	logger = TrainingLogger(
		log_dir=log_dir,
		use_tensorboard=cfg.get('enable_tensorboard', True),
		clear_existing=False,
		log_mcts_samples=False,
		config=cfg,
	)

	# meta.json + 既存CSV から累積アップデート数を復元（CSV優先）
	meta = _load_meta(data_dir)
	meta_updates = int(meta.get('train_updates_cumulative', 0) or 0)
	episodes_cumulative = int(meta.get('episodes_cumulative', 0) or 0)
	# last_checkpoint_episode を meta.json から取得（存在しない場合は0）
	if 'last_checkpoint_episode' not in meta:
		meta['last_checkpoint_episode'] = 0
	# CSVの最終 update_step を取得
	last_csv_step = 0
	try:
		csv_path = os.path.join(log_dir, 'train_updates.csv')
		if os.path.isfile(csv_path):
			with open(csv_path, 'r', encoding='utf-8') as rf:
				lines = rf.readlines()
			for line in reversed(lines):
				s = line.strip()
				if not s or s.startswith('update_step'):
					continue
				first = s.split(',')[0].strip()
				try:
					last_csv_step = int(first)
					break
				except Exception:
					continue
	except Exception:
		last_csv_step = 0
	train_updates_cum = max(meta_updates, last_csv_step)
	# ロガーのカウンタへ適用
	logger.update_step = int(train_updates_cum)

	# モデル/エージェント/環境生成前に共有リプレイバッファを用意
	shared_rb = ReplayBuffer(maxlen=cfg.get('buffer_size', 50000), path=None)
	# 直列モードではリプレイ縮小サイクルを無効化（設定・ロガー登録もしない）


	# --- Active File Pool (modular) ---
	all_files, pattern_attempts = _scan_selfplay_files(data_dir)
	found_count = len(all_files)
	pool_size = int(cfg.get('active_file_pool_size', 0) or 0)
	refresh_every = int(cfg.get('active_file_refresh_every_updates', 0) or 0)
	refresh_fraction = float(cfg.get('active_file_pool_refresh_fraction', 0.0) or 0.0)
	newest_bias = float(cfg.get('active_file_newest_bias', 0.5) or 0.5)
	
	# --- Buffer State スライディングウィンドウ方式（過学習防止）---
	# buffer_window_size > 0 の場合、buffer_state.jsonを使ってファイルリストを永続化し、
	# 同じデータの繰り返し学習を防止する
	buffer_window_size = int(cfg.get('buffer_window_size', 0) or 0)
	buffer_state_used = False
	used_files_for_session = []  # このセッションで使用したファイルを記録
	
	if buffer_window_size > 0 and not files_override:
		# Buffer State モード: 永続化されたファイルリストを使用
		buffer_state = _load_buffer_state(data_dir)
		prev_files = buffer_state.get('active_files', [])
		used_files = buffer_state.get('used_files', [])
		
		# 改善案B: ランダム選択モード（設定で有効化可能）
		enable_random_selection = bool(cfg.get('buffer_window_random_selection', True))
		
		# スライディングウィンドウでファイルリストを更新（改善案A・B対応）
		active_files, removed_files = _update_buffer_window(
			prev_files, all_files, buffer_window_size,
			used_files=used_files,
			enable_random_selection=enable_random_selection,
			logger=logger
		)
		buffer_state_used = True
		
		# このセッションで使用するファイルを記録（改善案A: 次回除外するため）
		# 注意: used_files_for_session は学習が実際に実行された場合のみ設定される
		# （学習ループ後の _save_buffer_state 呼び出し時に設定）
		used_files_for_session = []
		
		if logger:
			new_count = len([f for f in active_files if f not in set(prev_files)])
			logger.log_text(
				f"[buffer-state] mode=sliding_window window_size={buffer_window_size} "
				f"found={found_count} prev={len(prev_files)} new={new_count} current={len(active_files)} "
				f"random_selection={enable_random_selection} used_files_count={len(used_files)}"
			)
	else:
		# 従来モード: newest_biasに基づくファイル選択
		active_files = _select_initial_active_files(all_files, cfg, max_files=max_files, files_override=files_override)
		# ファイル選択のログ出力（newest_bias情報を含む）
		if logger and active_files:
			newest_count = int(len(active_files) * newest_bias)
			random_count = len(active_files) - newest_count
			logger.log_text(f"[file-selection] found={found_count} selected={len(active_files)} newest={newest_count} random={random_count} newest_bias={newest_bias:.2f}")
	eff_max_samp0 = None
	try:
		eff_max_samp0 = max_samples_per_file if (max_samples_per_file is not None and max_samples_per_file > 0) else (cfg.get('ingest_max_samples_per_file') or None)
	except Exception:
		pass
	total_train_all = 0
	total_val_all = 0
	preloaded: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
	# By default, initial_active_files == active_files. If resume_defer_preload is
	# enabled we will only preload a small recent slice to avoid long startup I/O.
	# ただし、buffer_stateモード時は全ファイルを読み込むため制限しない
	initial_active_files = list(active_files) if active_files is not None else []
	try:
		# buffer_stateモード時は全ファイルを読み込む（制限なし）
		if buffer_state_used:
			initial_active_files = list(active_files) if active_files is not None else []
			if logger:
				logger.log_text(f"[buffer-state] loading all {len(initial_active_files)} files (no defer)")
		# Optionally defer preloading heavy sample files after resume.
		# If `resume_defer_preload` is True in config, skip loading files here to avoid long blocking I/O.
		elif bool(cfg.get('resume_defer_preload', True)):
			# limit initial load to newest N files to avoid long blocking I/O at resume
			try:
				# 動的ファイル数計算を試行
				dynamic_result = _compute_dynamic_file_and_sample_params(cfg, train_updates_cum, len(active_files) if active_files else 0, logger)
				if dynamic_result is not None:
					n, _ = dynamic_result  # ファイル数のみ使用
				else:
					n = cfg.get('resume_initial_load_files')
					if n is None:
						# デフォルトはingest_max_filesを使用
						n = int(cfg.get('ingest_max_files') or 40)
					else:
						n = int(n)
					if n < 1:
						n = 1
			except Exception:
				n = 40
			# choose newest N from active_files (active_files is sorted ascending)
			# Prefer buffer_state.json active_files if present (useful when buffer/window managed externally)
			bs_path = os.path.join(data_dir, 'buffer_state.json')
			initial_active_files = []
			try:
				if os.path.isfile(bs_path):
					with open(bs_path, 'r', encoding='utf-8') as _bf:
						_bs = json.load(_bf) or {}
						bs_active = _bs.get('active_files') or []
						# filter to existing files
						bs_active = [f for f in bs_active if os.path.isfile(f)]
						if bs_active:
							# if bs_active larger than n, take the newest n
							initial_active_files = list(bs_active[-n:]) if n and len(bs_active) > n else list(bs_active)
							# Ensure the session's active_files uses the buffer_state canonical list
							try:
								active_files = list(bs_active)
							except Exception:
								pass
				# fallback to active_files selection if buffer_state absent or empty
				if not initial_active_files:
					if active_files:
						initial_active_files = list(active_files[-n:])
					else:
						initial_active_files = []
			except Exception:
				# on any error fallback to existing behavior
				if active_files:
					initial_active_files = list(active_files[-n:])
				else:
					initial_active_files = []
			if logger:
				logger.log_text(f"[INFO] resume_defer_preload enabled: limiting initial preload to latest {len(initial_active_files)} files (found_count={found_count})")
		else:
			initial_active_files = list(active_files) if active_files is not None else []
			for _fp in (initial_active_files or []):
				_parts_all = _load_samples_from_file(_fp, max_samples=(int(eff_max_samp0) if eff_max_samp0 else None))
				preloaded[_fp] = _parts_all
				try:
					total_train_all += len(_parts_all.get('train', []) or [])
					total_val_all += len(_parts_all.get('val', []) or [])
				except Exception:
					pass
			if logger:
				counts_repr = ' '.join([f"{pat}={len(lst)}" for pat,lst in pattern_attempts])
				logger.log_text(f"[INFO] glob_scan data_dir={data_dir} {counts_repr} found_count={found_count} pool_init_count={len(active_files)} initial_load_count={len(initial_active_files)} total_train_sel={total_train_all} total_val_sel={total_val_all} pool_strategy={'active' if pool_size>0 else 'legacy'}")
	except Exception:
		pass

	# Prepare initial split using helper
	try:
		_seed = cfg.get('seed')
	except Exception:
		_seed = None
	val_ratio = float(cfg.get('val_split_ratio', 0.1) or 0.1)
	if val_ratio < 0.0: val_ratio = 0.0
	if val_ratio > 0.9: val_ratio = 0.9
	try:
		# 動的サンプルサイジング: total_updates に基づいてサンプル数を調整
		# buffer_state_used が True でも fixed_total_samples を優先して制限を適用
		dynamic_result = _compute_dynamic_file_and_sample_params(cfg, train_updates_cum, len(initial_active_files), logger)
		if dynamic_result is not None:
			_, eff_max_samp = dynamic_result  # サンプル数のみ使用（ファイル数は既に適用済み）
		else:
			eff_max_samp = max_samples_per_file if (max_samples_per_file is not None and max_samples_per_file > 0) else (cfg.get('ingest_max_samples_per_file') or None)
		if eff_max_samp: eff_max_samp = int(eff_max_samp)
		
		if logger:
			if buffer_state_used:
				logger.log_text(f"[buffer-state] applying sample limit: eff_max_samp={eff_max_samp}")
			else:
				logger.log_text(f"[sampling] eff_max_samp={eff_max_samp}")
	except Exception:
		eff_max_samp = None
	# Use initial_active_files for the first rebuild to limit startup I/O when resume_defer_preload is set.
	# Instrumentation: measure durations of sample loading and replay rebuild to diagnose long resume pauses.
	try:
		if logger:
			logger.flush_buffers(force=True)
	except Exception:
		pass
	# サンプルロードを非同期で開始し、モデル初期化と並行させる
	load_future = None
	load_executor = None
	t_load_start = None
	train_part = []
	val_part = []
	train_samples_total = 0
	val_samples_total = 0
	initial_val_files = set()
	if initial_active_files:
		import concurrent.futures as _fut
		t_load_start = time.time()
		if logger:
			try:
				logger.log_text(f"[resume-timer] before_load_samples files={len(initial_active_files)}")
				logger.flush_buffers(force=True)
			except Exception:
				pass
		_log_memory_usage(logger, cfg, 'before_load_samples')
		load_executor = _fut.ThreadPoolExecutor(max_workers=1)
		load_future = load_executor.submit(
			_load_samples_for_files,
			initial_active_files,
			max_samples_per_file=eff_max_samp,
			seed=_seed,
			val_ratio=val_ratio,
			logger=logger,
			cfg=cfg,
			preloaded=preloaded,
		)

	# モデル/エージェント/環境生成 (今回は永続リプレイを使わずエフェメラル)
	bundle = create_env_and_agents(
		cfg,
		context='main',
		model_path=cfg.get('checkpoint_path'),
		shared_replay=shared_rb,
		logger=logger,
	)

	learner: AlphaZeroAgent = bundle.agents[cfg.get('learning_player_id', 0)]

	# Log opponent assignment summary (which players use latest/past/rule)
	try:
		lines = []
		for ag in bundle.agents:
			pid = getattr(ag, 'player_id', None)
			atype = type(ag).__name__
			# prefer explicit flag if present
			is_latest = None
			try:
				is_latest = bool(getattr(ag, 'is_using_latest_model', False))
			except Exception:
				is_latest = None
			# checkpoint identifier if available
			ck = getattr(ag, 'checkpoint_name', None)
			# compare model identity to bundle.model
			same_as_bundle = False
			try:
				same_as_bundle = (id(getattr(ag, 'model', None)) == id(getattr(bundle, 'model', None)))
			except Exception:
				same_as_bundle = False
			if is_latest is True:
				status = 'latest'
			elif is_latest is False:
				status = 'past'
			else:
				# fallback based on class
				status = 'rule' if atype.lower().startswith('rule') else ('latest' if same_as_bundle else 'unknown')
			lines.append(f"player={pid} type={atype} status={status} checkpoint={ck} same_as_bundle={same_as_bundle}")
		msg = "[arena-init] " + "; ".join(lines)
		if logger:
			logger.log_text(msg)
		else:
			print(msg)
	except Exception:
		pass

	# モデル再開ログ（ロード成功時のみ）
	try:
		model_path = cfg.get('checkpoint_path') or os.path.join(cfg.get('checkpoint_dir', 'checkpoints'), 'policy_value_latest.pt')
		if bool(getattr(bundle, 'loaded_from_checkpoint', False)) and os.path.isfile(model_path):
			logger.log_text(f"[resume] loaded model from {model_path}")
	except Exception:
		pass

	# optimizer / scheduler 再開
	try:
		opt_p = os.path.join(cfg['checkpoint_dir'], 'optimizer_latest.pt')
		# If model checkpoint exists, try a permissive model reload (strict=False)
		# and detect param-key mismatches. If the model structure changed, recreate optimizer
		# instead of loading possibly-incompatible optimizer.state_dict().
		model_path = cfg.get('checkpoint_path') or os.path.join(cfg.get('checkpoint_dir', 'checkpoints'), 'policy_value_latest.pt')
		mismatch_detected = False
		if model_path and os.path.isfile(model_path):
				import torch as _torch
				# load ckpt meta (weights_only if available)
				try:
					ck = _torch.load(model_path, map_location='cpu', weights_only=True)
				except TypeError:
					ck = _torch.load(model_path, map_location='cpu')
				state_dict = None
				if isinstance(ck, dict) and 'state_dict' in ck:
					state_dict = ck.get('state_dict')
				elif isinstance(ck, dict):
					state_dict = ck
				# Try permissive load into current model to allow head changes
				if state_dict is not None:
					try:
						missing_keys, unexpected_keys = learner.model.load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]
						
						# If new parameters were added (missing_keys), reinitialize them properly
						if missing_keys:
							logger.log_text(f"[resume] missing keys detected (新しいパラメータを再初期化): {missing_keys}")
							# Reinitialize missing parameters
							for key in missing_keys:
								try:
									# Get the parameter/buffer
									parts = key.split('.')
									module = learner.model
									for part in parts[:-1]:
										module = getattr(module, part)
									param_name = parts[-1]
									param = getattr(module, param_name)
									
									# Reinitialize with proper initialization (not zero)
									if 'weight' in param_name:
										_torch.nn.init.xavier_uniform_(param)
										logger.log_text(f"[resume] reinitialized {key} with xavier_uniform")
									elif 'bias' in param_name:
										_torch.nn.init.zeros_(param)
								except Exception as e:
									logger.log_text(f"[resume] failed to reinitialize {key}: {e}")
						
						# Detect already-present parameters that are all-zero (common after struct changes)
						critical_params = [
							'self_encoder.0.weight',
							'context_encoder.0.weight',
							'policy_head.0.weight',
							'value_head.0.weight',
							'hand_head.0.weight',
						]
						for key in critical_params:
							if key in learner.model.state_dict():
								try:
									param = learner.model.state_dict()[key]
									if hasattr(param, 'abs') and float(param.abs().max().item()) < 1e-8:
										parts = key.split('.')
										module = learner.model
										for part in parts[:-1]:
											module = getattr(module, part)
										param_name = parts[-1]
										p = getattr(module, param_name)
										_torch.nn.init.xavier_uniform_(p)
										logger.log_text(f"[resume] reinitialized zeroed param: {key}")
								except Exception as e:
									logger.log_text(f"[resume] failed to inspect/reinit {key}: {e}")
						
						if unexpected_keys:
							logger.log_text(f"[resume] unexpected keys (古いパラメータを無視): {unexpected_keys}")
					except Exception as e:
						# best-effort permissive load failed -> ignore and continue
						logger.log_text(f"[resume] model load failed: {e}")
				try:
					model_keys = set(list(learner.model.state_dict().keys()))
					ck_keys = set(list(state_dict.keys())) if state_dict is not None else set()
					if model_keys != ck_keys:
						mismatch_detected = True
				except Exception:
					mismatch_detected = False

		if os.path.isfile(opt_p):
			if hasattr(learner, 'ensure_optimizer'):
				# If model mismatched, recreate optimizer (ensure) but skip loading state
				if mismatch_detected:
					learner.ensure_optimizer()
					logger.log_text(f"[resume] model-param mismatch detected, recreated optimizer instead of loading {opt_p}")
					ok = False
				else:
					learner.ensure_optimizer()
					ok = False
					if hasattr(learner, 'load_optimizer'):
						ok = bool(learner.load_optimizer(opt_p, map_location=bundle.device))
			
	except Exception as e:
		logger.log_text(f"[WARN] optimizer load failed: {e}")
	# スケジューラの読み込みは学習率同期コードで行うため、ここではスキップ
	# (load_scheduler() が _update_step を設定してしまうため、学習率同期コードと競合する)
	try:
		sch_p = os.path.join(cfg['checkpoint_dir'], 'scheduler_latest.pt')
		# スケジューラファイルが存在しても、学習率同期コードで再構築するため読み込まない
		# if os.path.isfile(sch_p):
		# 	if hasattr(learner, 'ensure_scheduler'):
		# 		learner.ensure_scheduler()
		# 	ok = False
		# 	if hasattr(learner, 'load_scheduler'):
		# 		ok = bool(learner.load_scheduler(sch_p))
		pass
	except Exception as e:
		logger.log_text(f"[WARN] scheduler load failed: {e}")

	# --- 学習率スケジューラの同期修正 ---
	# optimizer.load_state_dict() が古いlrを復元するため、
	# schedulerを累積ステップ数に基づいて正しく再初期化し、optimizer lrを更新する
	# 注意: train_updates_cum == 0 の場合も、古いチェックポイントから誤ったlrがロードされる可能性があるため、
	# 常にlrを正しい値に設定する
	
	# _update_stepの設定はoptimizerの有無に関係なく実行（学習ステップカウントの整合性を保つため）
	step = int(train_updates_cum)
	if not hasattr(learner, '_update_step'):
		learner._update_step = 0
	learner._update_step = step  # train_updates_cum に基づく正しいステップ数に設定
	logger.log_text(f"[lr-sync] set learner._update_step = {step} (train_updates_cum)")
	
	try:
		# Optimizerが存在することを保証
		if not hasattr(learner, '_optimizer') or learner._optimizer is None:
			if hasattr(learner, 'ensure_optimizer'):
				learner.ensure_optimizer()
				logger.log_text(f"[lr-sync] ensured optimizer exists")
		
		if hasattr(learner, '_optimizer') and learner._optimizer is not None:
			import math as _math
			base_lr = float(cfg.get('lr', 1e-4))
			lr_min = float(cfg.get('lr_min', 1e-5))
			warmup = int(cfg.get('lr_warmup_steps', 0) or 0)
			tmax = int(cfg.get('lr_cosine_T_max_updates', 0) or 0)
			min_scale = (lr_min / base_lr) if base_lr > 0 else 0.0
			
			# 累積ステップ数に基づく正しい学習率を計算
			step = int(train_updates_cum)
			if warmup > 0 and step < warmup:
				lr_scale = max(1e-8, float(step + 1) / float(warmup))
			elif tmax > warmup and tmax > 0:
				prog = min(1.0, float(step - warmup) / float(max(1, tmax - warmup)))
				cos_factor = 0.5 * (1.0 + _math.cos(_math.pi * prog))
				lr_scale = min_scale + (1.0 - min_scale) * cos_factor
			else:
				lr_scale = min_scale
			
			correct_lr = base_lr * lr_scale
			# Value Head用のスケール
			value_head_lr_scale_cfg = float(cfg.get('value_head_lr_scale', 1.0))
			
			# スケジューラの状態確認と同期
			sched = getattr(learner, '_scheduler', None)
			need_recreate = False
			
			if sched is not None:
				# 既存のスケジューラのlast_epochをチェック
				current_last_epoch = getattr(sched, 'last_epoch', None)
				expected_last_epoch = step - 1  # 次のstep()でstepになる想定
				# last_epochが大きくずれている、またはNoneの場合は再作成
				if current_last_epoch is None or abs(current_last_epoch - expected_last_epoch) > 2:
					need_recreate = True
					logger.log_text(f"[lr-sync] scheduler last_epoch mismatch: current={current_last_epoch} expected={expected_last_epoch}, recreating")
			
			if need_recreate or sched is None:
				# スケジューラを（再）作成
				if hasattr(learner, '_scheduler'):
					learner._scheduler = None
				if hasattr(learner, 'ensure_scheduler'):
					learner.ensure_scheduler()
					sched = getattr(learner, '_scheduler', None)
					if sched is not None:
						# 累積ステップに合わせてlast_epochを補正
						sched.last_epoch = step - 1
						logger.log_text(f"[lr-sync] scheduler created/reset with last_epoch={getattr(sched, 'last_epoch', None)} (expected {step-1})")
			
			# スケジューラが正常に存在する場合、学習率を更新
			# 注意: scheduler.step()はoptimizer.step()の後に呼ばれるべきなので、
			# ここではlast_epochを設定するだけで、実際のstep()はtrain_step内で実行される
			if sched is not None:
				try:
					# last_epochを累積ステップに揃える（step()は呼ばない）
					sched.last_epoch = step - 1
					# 学習率を手動で設定（scheduler.step()の代わり）
					# これにより、optimizer.step()の前にscheduler.step()が呼ばれることを防ぐ
					for idx, group in enumerate(learner._optimizer.param_groups):
						if group.get('name') == 'value_head':
							group['lr'] = correct_lr * value_head_lr_scale_cfg
						else:
							group['lr'] = correct_lr
					current_lr = learner._optimizer.param_groups[0]['lr'] if learner._optimizer else None
					lr_str = f"{current_lr:.10e}" if current_lr is not None else "None"
					logger.log_text(f"[lr-sync] scheduler last_epoch set to {getattr(sched, 'last_epoch', None)}, lr manually set to {lr_str}")
				except Exception as e:
					logger.log_text(f"[lr-sync] scheduler setup failed: {e}")
			else:
				# スケジューラがない場合のみ手動でlrを設定
				logger.log_text(f"[lr-sync] no scheduler, setting lr manually")
				for idx, group in enumerate(learner._optimizer.param_groups):
					if group.get('name') == 'value_head':
						group['lr'] = correct_lr * value_head_lr_scale_cfg
					else:
						group['lr'] = correct_lr
			
			logger.log_text(f"[lr-sync] synced lr to step={step}: correct_lr={correct_lr:.10e} lr_scale={lr_scale:.6f}")
	except Exception as e:
		logger.log_text(f"[WARN] lr-sync failed: {e}")
	# 非同期ロード完了を待機
	t_load_end = t_load_start
	if load_future is not None:
		try:
			train_part, val_part, train_samples_total, val_samples_total, initial_val_files = load_future.result()
			t_load_end = time.time()
		except Exception:
			# 失敗時も以降の処理は継続
			train_part = train_part or []
			val_part = val_part or []
		finally:
			if load_executor is not None:
				load_executor.shutdown(wait=False)
							
	if t_load_end is None:
		t_load_end = time.time()

	# メモリ効率化: preloaded辞書をクリア（_load_samples_for_filesで使用済みなので不要）
	try:
		if 'preloaded' in locals() and preloaded is not None:
			preloaded.clear()
			del preloaded
	except Exception:
		pass
	# デバッグ: 読み込まれたサンプルにvalue_qが含まれているか確認
	if bool(cfg.get('debug_value_mix', False)) and train_part:
		try:
			from agents.agent_utills.training import _extract_value_q
			q_count = sum(1 for s in train_part[:min(100, len(train_part))] if _extract_value_q(s) is not None)
			player_id_count = sum(1 for s in train_part[:min(100, len(train_part))] if s.get('player_id') is not None)
			print(f"[DEBUG_VALUE_MIX] train_part size={len(train_part)}, samples_with_q (first 100)={q_count}, samples_with_player_id (first 100)={player_id_count}")
		except Exception as e:
			print(f"[DEBUG_VALUE_MIX] failed to check train_part: {e}")
	
	# ロード時間の計測結果をログ出力
		if logger:
			logger.log_text(
				f"[resume-timer] load_samples took={(t_load_end - t_load_start) if t_load_start is not None and t_load_end is not None else 0.0:.3f}s "
				f"files={len(initial_active_files)} samples_train={train_samples_total} samples_val={val_samples_total}"
			)
			logger.flush_buffers(force=True)
	_log_memory_usage(logger, cfg, 'after_load_samples')
	_maybe_collect_gc(cfg)
	if bool(cfg.get('aggressive_gc', False)):
		_log_memory_usage(logger, cfg, 'after_load_samples_gc')
		
	# rebuild replay with timing
	t_rebuild_start = time.time()
	chunk_sz = int(cfg.get('rebuild_chunk_size', 0) or 0)
	train_samples_total, val_samples_total = _rebuild_replay(shared_rb, train_part, val_part, logger, chunk_size=chunk_sz)
	t_rebuild_end = time.time()
	# メモリ効率化: train_partとval_partを明示的にクリア（ReplayBufferに追加済みなので不要）
	try:
		del train_part
		del val_part
	except Exception:
		pass
	_maybe_collect_gc(cfg)
	if bool(cfg.get('aggressive_gc', False)):
		_log_memory_usage(logger, cfg, 'after_rebuild_gc')
	# デバッグ: リプレイバッファに追加されたサンプルにvalue_qが含まれているか確認
	if bool(cfg.get('debug_value_mix', False)) and len(shared_rb) > 0:
		try:
			from agents.agent_utills.training import _extract_value_q
			# ReplayBufferからサンプルを取得（iter_allまたは直接アクセス）
			if hasattr(shared_rb, 'iter_all'):
				rb_samples_all = list(shared_rb.iter_all(owner_pid=None))[:min(100, len(shared_rb))]
				rb_samples_pid0 = list(shared_rb.iter_all(owner_pid=0))[:min(100, len(shared_rb))]
			else:
				rb_samples_all = list(shared_rb)[:min(100, len(shared_rb))]
				rb_samples_pid0 = []
			q_count_rb_all = sum(1 for s in rb_samples_all if _extract_value_q(s) is not None)
			q_count_rb_pid0 = sum(1 for s in rb_samples_pid0 if _extract_value_q(s) is not None)
			player_id_count_rb = sum(1 for s in rb_samples_all if s.get('player_id') is not None)
			print(f"[DEBUG_VALUE_MIX] replay_buffer size={len(shared_rb)}, samples_with_q (all, first 100)={q_count_rb_all}, samples_with_q (pid=0, first 100)={q_count_rb_pid0}, samples_with_player_id (first 100)={player_id_count_rb}")
		except Exception as e:
			print(f"[DEBUG_VALUE_MIX] failed to check replay_buffer: {e}")
	
		if logger:
			# _rebuild_replayの返り値を使用（再カウント不要）
			logger.log_text(f"[resume-timer] rebuild_replay took={t_rebuild_end - t_rebuild_start:.3f}s train={train_samples_total} val={val_samples_total} total={len(shared_rb)}")
			logger.flush_buffers(force=True)
		
	t_rebuild_end = time.time()
	try:
		if logger:
			# _rebuild_replayの返り値を使用（再カウント不要）
			logger.log_text(f"[resume-timer] rebuild_replay took={t_rebuild_end - t_rebuild_start:.3f}s train={train_samples_total} val={val_samples_total} total={len(shared_rb)}")
			logger.flush_buffers(force=True)
	except Exception:
		pass
	_log_memory_usage(logger, cfg, 'after_rebuild')
	# 集約進捗表示 (episodes は meta から取得した累積値)
	# replay_size は実際のリプレイバッファ長を表示する
	try:
		_total_samples_all = total_train_all + total_val_all
	except Exception:
		_total_samples_all = len(shared_rb)
	logger.log_text(f"[status-train] ep_done={episodes_cumulative} active_files={len(active_files)} appended_train={train_samples_total} appended_val={val_samples_total} replay_size={len(shared_rb)} split_ratio={1.0 - val_ratio:.2f}/{val_ratio:.2f} updates_before={train_updates_cum}")

	# 事前検証: 最初の学習行から val_* を必ず埋めるため、ここで一度実施
	# If `resume_skip_prevalidation` is True, skip this potentially heavy validation after resume.
	if bool(cfg.get('resume_skip_prevalidation', False)):
		if logger:
			logger.log_text('[INFO] resume_skip_prevalidation enabled: skipping pre-train validation')
	else:
		_run_validation(learner, logger, cfg, note='pre-train')

	# 学習ループ
	# updates is already converted to int at the beginning of the function
	val_every = int(cfg.get('val_eval_every_updates', 0) or 0)
	batch_size = int(cfg.get('batch_size', 256) or 256)
	version_interval = int(version_interval or 0)
	model_version = int(meta.get('model_version', 0) or 0)  # self-play と共有する番号を継続利用
	# CSVログ出力間隔（csv_train_log_every）を使用 - 検証とログを同じタイミングで実行
	csv_log_interval = int(cfg.get('csv_train_log_every', 50) or 50)
	log_interval = csv_log_interval
	# CSVには検証結果も含めて出力したいため、log_intervalごとに検証も実行する

	# ベストモデル保存用 (打ち切りは行わない)
	early_min_delta = float(cfg.get('early_stop_min_delta', 0.0) or 0.0)
	early_gap_delta = float(cfg.get('early_stop_gap_min_delta', 0.0) or 0.0)
	early_best: tuple[float | None, float | None] | None = None  # (val_value_loss, gap)
	early_best_path = os.path.join(cfg.get('checkpoint_dir', 'checkpoints'), 'policy_value_best.pt')
	# In-memory snapshot of best model state_dict to avoid repeated disk writes
	_best_state_dict_snapshot = None

	last_loss_info = None
	# hand head warmup: 設定で指定された最初の N 更新の間だけ hand_pred_loss_coef を一時上書き
	hand_warmup_updates = int(cfg.get('hand_warmup_updates', 0) or 0)
	hand_warmup_coef = cfg.get('hand_warmup_coef', None)
	warmup_end = None
	original_hand_coef = None
	if hand_warmup_updates > 0 and hand_warmup_coef is not None:
		warmup_end = int(train_updates_cum + hand_warmup_updates)
		try:
			original_hand_coef = float(cfg.get('hand_pred_loss_coef', 0.0) or 0.0)
		except Exception:
			original_hand_coef = 0.0
		# resume 時に既にウォームアップ期間内であれば即時反映
		if train_updates_cum < warmup_end:
			try:
				cfg['hand_pred_loss_coef'] = float(hand_warmup_coef)
				if logger:
					logger.log_text(f"[hand-warmup] enabled until update={warmup_end} hand_pred_loss_coef={hand_warmup_coef} (original={original_hand_coef})")
			except Exception:
				pass
	# --- 学習ループ + プールリフレッシュ ---
	# instrumentation timers (aggregate per-log-interval)
	_t_train_acc = 0.0
	_t_val_acc = 0.0
	_t_refresh_acc = 0.0
	_t_misc_acc = 0.0
	_t_interval_start = time.time()
	for i in range(updates):
		# current absolute update index (cumulative)
		current_step = train_updates_cum + i + 1
		# Warmup期の適用/解除を毎ステップチェックして learner.config に反映
		if warmup_end is not None:
			if current_step <= warmup_end:
				try:
					learner.config['hand_pred_loss_coef'] = float(hand_warmup_coef)
				except Exception:
					pass
			else:
				# ウォームアップ終了時に元の係数へ復元
				if original_hand_coef is not None:
					try:
						# only restore once
						if float(learner.config.get('hand_pred_loss_coef', 0.0) or 0.0) != float(original_hand_coef):
							learner.config['hand_pred_loss_coef'] = float(original_hand_coef)
							if logger:
								logger.log_text(f"[hand-warmup] ended at update={current_step-1}, restored hand_pred_loss_coef={original_hand_coef}")
					except Exception:
						pass
		# measure train_step time
		t0 = time.time()
		loss_info = learner.train_step(batch_size=batch_size)
		t1 = time.time()
		t_train = t1 - t0
		_t_train_acc += t_train
		_maybe_collect_gc(cfg)
		
		# 50ステップごとに勾配ノルムを表示
		current_step = train_updates_cum + i + 1
		if (current_step % 300) == 0:
			try:
				import math as _m
				import torch as _t
				
				grad_norm = None
				value_head_grad_norm = None
				policy_head_grad_norm = None
				hand_head_grad_norm = None
				
				if isinstance(loss_info, dict):
					grad_norm = loss_info.get('grad_norm')
					value_head_grad_norm = loss_info.get('value_head_grad_norm')
				
				# モデルから直接各ヘッドの勾配ノルムを計算
				if learner.model is not None:
					try:
						# Policy Headの勾配ノルム
						policy_sq = 0.0
						policy_count = 0
						for name, param in learner.model.named_parameters():
							if 'policy_head' in name and param.requires_grad and param.grad is not None:
								policy_sq += float(param.grad.detach().data.norm(2).item() ** 2)
								policy_count += 1
						if policy_count > 0:
							policy_head_grad_norm = float(_m.sqrt(policy_sq))
					except Exception:
						pass
					
					try:
						# Hand Headの勾配ノルム
						hand_sq = 0.0
						hand_count = 0
						for name, param in learner.model.named_parameters():
							if 'hand_head' in name and param.requires_grad and param.grad is not None:
								hand_sq += float(param.grad.detach().data.norm(2).item() ** 2)
								hand_count += 1
						if hand_count > 0:
							hand_head_grad_norm = float(_m.sqrt(hand_sq))
					except Exception:
						pass
				
				# ログ出力（ヘッド別勾配ノルム）
				try:
					def _fmt(v):
						return f"{v:.6f}" if v is not None else "None"
					msg = (
						f"[grad-norm-head] step={current_step} "
						f"total={_fmt(grad_norm)} policy={_fmt(policy_head_grad_norm)} "
						f"value={_fmt(value_head_grad_norm)} hand={_fmt(hand_head_grad_norm)}"
					)
					if logger:
						logger.log_text(msg)
				except Exception:
					pass
			except Exception as e:
				# エラーは無視して続行
				pass
		
		if ((i + 1) % log_interval) == 0 or (i + 1) == updates:
			_log_memory_usage(logger, cfg, f"train_loop_step={train_updates_cum + i + 1}")
		# 空データ回避ログ
		if isinstance(loss_info, dict) and loss_info.get('loss') is None and loss_info.get('reason') == 'no_data':
			# リプレイバッファの状態を詳細にログ出力
			try:
				rb_size = len(shared_rb) if shared_rb else 0
				# 学習プレイヤー（player_id=0）のサンプルを確認
				if hasattr(shared_rb, 'iter_all'):
					all_samples_pid0 = list(shared_rb.iter_all(owner_pid=0))
					train_samples_pid0 = [s for s in all_samples_pid0 if isinstance(s, dict) and s.get('split') != 'val']
					has_value_samples_pid0 = [s for s in train_samples_pid0 if s.get('value') is not None]
					logger.log_text(
						f'[warn] train_step skipped (no_data) '
						f'replay_buffer_size={rb_size} '
						f'player_id=0_total={len(all_samples_pid0)} '
						f'player_id=0_train={len(train_samples_pid0)} '
						f'player_id=0_train_with_value={len(has_value_samples_pid0)}'
					)
				else:
					all_samples = list(shared_rb) if shared_rb else []
					train_samples = [s for s in all_samples if isinstance(s, dict) and s.get('split') != 'val']
					has_value_samples = [s for s in train_samples if s.get('value') is not None]
					logger.log_text(
						f'[warn] train_step skipped (no_data) '
						f'replay_buffer_size={rb_size} '
						f'train_samples={len(train_samples)} '
						f'train_with_value={len(has_value_samples)}'
					)
			except Exception as e:
				logger.log_text(f'[warn] train_step skipped (no_data) - 詳細確認失敗: {e}')
		# アクティブファイルプール刷新（更新回数ベース）
		# sliding_window モード (buffer_window_size > 0) の場合はリフレッシュをスキップ
		if pool_size > 0 and refresh_every > 0 and buffer_window_size == 0 and ((train_updates_cum + i + 1) % refresh_every == 0):
			# measure pool refresh time
			t0r = time.time()
			try:
				active_files, _stats = _refresh_active_pool(active_files, all_files, cfg, shared_rb, val_ratio, _seed, eff_max_samp, logger, train_updates_cum=train_updates_cum + i + 1, data_dir=data_dir)
			except Exception as e:
				logger.log_text(f"[WARN] active-pool refresh failed: {e}")
			t1r = time.time()
			_t_refresh_acc += (t1r - t0r)

		# csv_train_log_every の間隔でのみ検証とCSVログ出力
		if ((i + 1) % log_interval) == 0:
			# 検証を実施（CSVに記録するため）
			# measure validation time
			t0v = time.time()
			val_metrics = _run_validation(learner, logger, cfg, batch_size=batch_size, note=f'update={train_updates_cum + i + 1}')
			t1v = time.time()
			_t_val_acc += (t1v - t0v)
			
			# ベストモデル更新判定 (val_value_loss 最小、同値なら Train/Val Gap 最小を優先)
			try:
				val_v = None
				if isinstance(val_metrics, dict):
					val_v = val_metrics.get('value_loss')
				train_v = None
				if isinstance(loss_info, dict):
					train_v = loss_info.get('value_loss') or loss_info.get('loss')
				gap = None
				if val_v is not None and train_v is not None:
					try:
						gap = abs(float(val_v) - float(train_v))
					except Exception:
						gap = None
				improved = False
				if val_v is not None:
					if early_best is None:
						improved = True
					else:
						best_val, best_gap = early_best
						if best_val is None or val_v < best_val - early_min_delta:
							improved = True
						elif best_val is not None and abs(val_v - best_val) <= early_min_delta:
							if gap is not None:
								if best_gap is None or gap < best_gap - early_gap_delta:
									improved = True
				if improved:
					early_best = (val_v, gap)
					# Snapshot model state_dict into memory (CPU tensors) to save once after training
					try:
						if hasattr(learner, 'model') and learner.model is not None:
							# copy to CPU and clone to detach from GPU tensors
							try:
								_best_state_dict_snapshot = {k: v.cpu().clone() for k, v in learner.model.state_dict().items()}
							except Exception:
								# fallback: shallow copy
								_best_state_dict_snapshot = dict(learner.model.state_dict())
					except Exception:
						_best_state_dict_snapshot = None
			except Exception:
				pass
			
			# CSVにログを出力（検証結果も明示的に含める）
			if isinstance(loss_info, dict) and loss_info.get('loss') is not None:
				try:
					loss_info.setdefault('train_count', train_updates_cum + i + 1)
					# 検証結果を明示的に loss_info に追加
					if isinstance(val_metrics, dict):
						loss_info['val_policy_loss'] = val_metrics.get('policy_loss')
						loss_info['val_value_loss'] = val_metrics.get('value_loss')
						loss_info['val_hand_pred_loss'] = val_metrics.get('hand_pred_loss')
						loss_info['val_hand_recall'] = val_metrics.get('hand_recall')
				except Exception:
					pass
				logger.log_train(loss_info)
				last_loss_info = loss_info
				# instrumentation: report timing for this interval
				try:
					interval_now = time.time()
					total_interval = interval_now - _t_interval_start
					logger.log_text(
						f"[time-prof] updates={train_updates_cum + i + 1 - (log_interval-1)}-{train_updates_cum + i + 1} "
						f"train={_t_train_acc:.3f}s val={_t_val_acc:.3f}s refresh={_t_refresh_acc:.3f}s other={_t_misc_acc:.3f}s total={total_interval:.3f}s"
					)
				except Exception:
					pass
				# reset accumulators for next interval
				_t_train_acc = 0.0
				_t_val_acc = 0.0
				_t_refresh_acc = 0.0
				_t_misc_acc = 0.0
				_t_interval_start = time.time()

		# 100更新ごとに optimizer/scheduler の lr を events.log へ出力
		try:
			step_now = int(train_updates_cum + i + 1)
			if (step_now % 100) == 0:
				# ensure optimizer exists and fetch param group learning rates
				if hasattr(learner, 'ensure_optimizer'):
					try:
						learner.ensure_optimizer()
					except Exception:
						pass
				opt = getattr(learner, '_optimizer', None)
				lrs = []
				if opt is not None and hasattr(opt, 'param_groups'):
					try:
						for g in opt.param_groups:
							try:
								lrs.append(float(g.get('lr', None)))
							except Exception:
								lrs.append(None)
					except Exception:
						lrs = []
				# scheduler debug info
				sched = getattr(learner, '_scheduler', None)
				sched_last_epoch = getattr(sched, 'last_epoch', None) if sched else None
				update_step_internal = getattr(learner, '_update_step', None)
				# config params for reference
				base_lr = float(cfg.get('lr', 1e-4))
				lr_min = float(cfg.get('lr_min', 1e-5))
				warmup = int(cfg.get('lr_warmup_steps', 0) or 0)
				tmax = int(cfg.get('lr_cosine_T_max_updates', 0) or 0)
				# theoretical lr at this step based on config
				import math
				if tmax > warmup and step_now >= warmup:
					prog = min(1.0, float(step_now - warmup) / float(max(1, tmax - warmup)))
					cos_factor = 0.5 * (1.0 + math.cos(math.pi * prog))
					min_scale = (lr_min / base_lr) if base_lr > 0 else 0.0
					theoretical_lr = base_lr * (min_scale + (1.0 - min_scale) * cos_factor)
				elif step_now < warmup:
					theoretical_lr = base_lr * max(1e-8, float(step_now + 1) / float(warmup))
				else:
					theoretical_lr = lr_min
				# format and log
				try:
					lr_repr = ','.join([('None' if v is None else f"{v:.10e}") for v in lrs]) if lrs else 'unknown'
					if logger:
						logger.log_text(
							f"[lr-debug] step={step_now} lrs=[{lr_repr}] "
							f"sched_last_epoch={sched_last_epoch} internal_update_step={update_step_internal} "
							f"theoretical_lr={theoretical_lr:.10e} "
							f"cfg: base_lr={base_lr:.6e} lr_min={lr_min:.6e} warmup={warmup} tmax={tmax}"
						)
				except Exception:
					pass
		except Exception:
			pass

	# 最後に最新モデル保存 (version_tag なし)
	# 保存前にモデルのパラメータを検証
	try:
		model = bundle.model
		if model is not None:
			state_dict = model.state_dict()
			critical_params = [
				'self_encoder.0.weight',
				'context_encoder.0.weight',
				'backbone_proj.weight',
				'backbone.0.lin1.weight',
				'backbone.0.lin2.weight',
				'policy_head.0.weight',
				'value_head.0.weight',
			]
			zero_params = []
			for key in critical_params:
				if key in state_dict:
					param = state_dict[key]
					if hasattr(param, 'abs'):
						abs_max = param.abs().max().item()
						if abs_max < 1e-8:
							zero_params.append(key)
			
			if zero_params:
				if logger:
					logger.log_text(f"[WARN] Before save_checkpoint: Found zero weight parameters: {zero_params[:5]}{'...' if len(zero_params) > 5 else ''}")
				print(f"[WARN] Before save_checkpoint: Found zero weight parameters: {zero_params[:5]}{'...' if len(zero_params) > 5 else ''}")
				# ゼロパラメータが見つかった場合、保存をスキップ
				if logger:
					logger.log_text(f"[WARN] Skipping checkpoint save due to zero parameters")
				print(f"[WARN] Skipping checkpoint save due to zero parameters")
				return
	except Exception as e:
		if logger:
			logger.log_text(f"[WARN] Parameter validation before save failed: {e}")
		print(f"[WARN] Parameter validation before save failed: {e}")
	
	_save_checkpoint(bundle, cfg, model_version, version_tag=None, logger=logger)

	# エピソード累計に基づくバージョン保存:
	# 10000, 20000, 30000... のちょうどの倍数でのみ1回保存する
	checkpoint_interval = int(version_interval or 0) if int(version_interval or 0) > 0 else int(cfg.get('checkpoint_interval_episodes', 10000) or 10000)
	last_saved_episode = int(meta.get('last_checkpoint_episode', 0) or 0)
	if checkpoint_interval > 0 and episodes_cumulative >= checkpoint_interval:
		# 現在到達している最大の倍数を計算（例: 30600 → 30000）
		target_multiple = (episodes_cumulative // checkpoint_interval) * checkpoint_interval
		# 前回保存した倍数より大きい倍数に到達した場合のみ保存
		# 例: last_saved=20000, target=30000 → 保存
		#     last_saved=30000, target=30000 → 保存しない（既に保存済み）
		if target_multiple > 0 and target_multiple > last_saved_episode:
			model_version += 1
			_save_checkpoint(bundle, cfg, model_version, version_tag=f'ep{target_multiple}', logger=logger)
			if logger:
				logger.log_text(
					f"[ckpt] saved version={model_version} at episode={target_multiple} (current_episodes={episodes_cumulative})"
				)
			# 保存した倍数を記録（次回はこれより大きい倍数に到達するまで保存しない）
			meta['last_checkpoint_episode'] = target_multiple
			try:
				_update_meta(data_dir, meta)
			except Exception:
				pass

	# 学習後の最終検証を一度実施（intervalに依らず終端の値を残す）
	_run_validation(learner, logger, cfg, batch_size=batch_size, note='post-train')
	# If we captured an in-memory best-state snapshot during training, save it once now.
	try:
		if _best_state_dict_snapshot is not None:
			from agents.models import PolicyValueNet
			# Reconstruct a model with same architecture meta as current bundle.model
			if bundle.model is not None:
				meta_model = bundle.model
				try:
					best_model = PolicyValueNet(
						max_policy_size=getattr(meta_model, 'max_policy_size', 128),
						hidden_size=getattr(meta_model, 'hidden_size', 128),
						num_players=getattr(meta_model, 'num_players', 4),
						use_full_features=getattr(meta_model, 'use_full_features', True),
						full_feature_dim=getattr(meta_model, 'full_feature_dim', None),
						enable_hand_prediction_head=getattr(meta_model, 'enable_hand_prediction_head', True),
						context_out_dim=getattr(meta_model, 'context_out_dim', 128),
					)
					# load snapshot
					best_model.load_state_dict(_best_state_dict_snapshot)
					# save best model to disk once
					os.makedirs(os.path.dirname(early_best_path), exist_ok=True)
					best_model.save(early_best_path, logger=logger, force_sync=True)  # type: ignore[arg-type]
				except Exception:
					# best-effort: try torch.save of state_dict if model construction fails
					try:
						ckpt = {'state_dict': _best_state_dict_snapshot}
						torch.save(ckpt, early_best_path)
					except Exception:
						if logger:
							logger.log_text(f"[save] WARNING: Failed to persist best-model snapshot: {_best_state_dict_snapshot is None}")
						pass
			# clear snapshot to free memory
			_best_state_dict_snapshot = None
	except Exception:
		pass
	_maybe_collect_gc(cfg)
	_log_memory_usage(logger, cfg, 'after_train_loop')

	train_updates_cum += updates
	# consumed_files は使用しない (毎回全ファイルを対象)
	meta['train_updates_cumulative'] = train_updates_cum
	meta['model_version'] = model_version
	meta['episodes_cumulative'] = episodes_cumulative  # 既存値保持 (self-play 側で更新済み想定)
	# last_checkpoint_episode は既にチェックポイント保存時に更新されているが、
	# 学習ループ終了時にも保持する（既存の値が上書きされないように）
	# meta は _load_meta で読み込まれているので、既存の last_checkpoint_episode は既に含まれている
	# チェックポイント保存時に更新された場合は新しい値が設定されている
	# ここでは明示的に設定する必要はないが、存在しない場合はデフォルト値0を設定
	if 'last_checkpoint_episode' not in meta:
		meta['last_checkpoint_episode'] = 0
	# リプレイサイズを meta に記録しておくと self-play 側から軽量に参照できる
	# meta へも全選択ファイル合計サンプル数を記録（学習に使った一部だけでなく全体規模を把握する目的）
	# リプレイサイズを meta に記録（表示/自己対局側参照用）。実メモリ内のバッファ長を採用。
	try:
		meta['replay_size'] = int(len(shared_rb))
	except Exception:
		pass
	logger.log_text(f"[summary-train] updates={updates} total_updates={train_updates_cum} model_version={model_version} replay_size={len(shared_rb)}")
	# Instrumentation: time meta update to detect long blocking IO during resume/finish
	t_meta_start = time.time()
	try:
		_update_meta(data_dir, meta)
	finally:
		t_meta_end = time.time()
		try:
			if logger:
				logger.log_text(f"[resume-timer] update_meta took={t_meta_end - t_meta_start:.3f}s")
				logger.flush_buffers(force=True)
		except Exception:
			pass
	
	# --- Buffer State 保存（スライディングウィンドウモード時）---
	# 次回学習時に同じファイルを再度読み込まないよう、現在のウィンドウ状態を永続化
	# 改善案A: 使用済みファイルを記録して次回除外
	# 注意: 学習が実際に実行された場合（updates > 0 かつ学習ループが実行された場合）のみ used_files を更新
	# 注意: updates <= 0 の場合の active_files 保存は早期リターン前で処理済み
	if buffer_state_used and updates > 0:
		try:
			# 学習が実行された場合のみ、使用したファイルを記録
			used_files_for_session = list(active_files)
			# Ensure active_files is filled up to window_size before persisting.
			try:
				desired = int(buffer_window_size)
			except Exception:
				desired = 0
			if desired > 0:
				try:
					# buffer_state may contain historical used_files
					prev_used = set(buffer_state.get('used_files', [])) if isinstance(buffer_state, dict) else set()
					# candidates: prefer unused files first
					candidates = [f for f in all_files if f not in active_files and f not in prev_used and os.path.isfile(f)]
					# sort by mtime ascending, pick newest later
					try:
						candidates.sort(key=lambda x: os.path.getmtime(x))
					except Exception:
						pass
					need = desired - len(active_files)
					if need > 0 and candidates:
						add = candidates[-need:]
						active_files = list(active_files) + add
						need = desired - len(active_files)
					# if still need, allow reusing used files (oldest-first)
					if need > 0:
						reuse = [f for f in all_files if f not in active_files and os.path.isfile(f)]
						try:
							reuse.sort(key=lambda x: os.path.getmtime(x))
						except Exception:
							pass
						if reuse:
							add2 = reuse[-need:]
							active_files = list(active_files) + add2
				# final clamp
					if len(active_files) > desired:
						active_files = active_files[-desired:]
				except Exception:
					# if anything fails, keep existing active_files
					pass
			# Also persist initial val selection as used to avoid reusing those files
			try:
				if initial_val_files:
					# remove initial val files from active list
					active_files = [f for f in active_files if f not in initial_val_files]
					_save_buffer_state(data_dir, active_files, used_files=list(initial_val_files), logger=logger)
				else:
					_save_buffer_state(data_dir, active_files, used_files=used_files_for_session, logger=logger)
			except Exception:
				# fallback to saving full used_files_for_session
				_save_buffer_state(data_dir, active_files, used_files=used_files_for_session, logger=logger)
		except Exception as e:
			if logger:
				logger.log_text(f"[WARN] buffer_state save failed: {e}")
	# 最終ロスのサマリーを明示出力
	if isinstance(last_loss_info, dict):
		try:
			loss = last_loss_info.get('loss')
			pl = last_loss_info.get('policy_loss') or last_loss_info.get('pl')
			vl = last_loss_info.get('value_loss') or last_loss_info.get('vl')
			# hand 取得キーを hand_pred_loss 優先へ修正 (過去 None 問題を解消)
			hand = last_loss_info.get('hand_pred_loss') or last_loss_info.get('hand_loss') or last_loss_info.get('hand')
			ent = last_loss_info.get('entropy') or last_loss_info.get('ent')
			kl = last_loss_info.get('kl')
			top1 = last_loss_info.get('top1') or last_loss_info.get('policy_top1_match')
			v_acc = last_loss_info.get('v_acc') or last_loss_info.get('value_acc')
			v_brier = last_loss_info.get('v_brier') or last_loss_info.get('value_brier')
			grad_n = last_loss_info.get('grad_norm')
			weight_n = last_loss_info.get('weight_norm')
			hand_t1 = last_loss_info.get('hand_top1')
			hand_t3 = last_loss_info.get('hand_top3')
			hand_b = last_loss_info.get('hand_brier')
			logger.log_text(f"[final-loss] loss={loss} pl={pl} vl={vl} hand={hand} ent={ent} kl={kl} top1={top1} v_acc={v_acc} v_brier={v_brier} grad_norm={grad_n} weight_norm={weight_n} hand_top1={hand_t1} hand_top3={hand_t3} hand_brier={hand_b}")
		except Exception:
			pass
	if hasattr(logger, 'flush_buffers'):
		logger.flush_buffers(force=True)

	# 永続リプレイ保存は無し (エフェメラル運用)

	print(f"[INFO] training finished updates={updates} total_updates={train_updates_cum}")


def main():
	parser = argparse.ArgumentParser(description='Non-parallel training (ingest self-play joblib files)')
	parser.add_argument('--data-dir', type=str, default='data', help='自己対局データディレクトリ')
	parser.add_argument('--log-dir', type=str, default='logs', help='ログディレクトリ (追記)')
	parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='チェックポイントディレクトリ')
	parser.add_argument('--model-path', type=str, default=None, help='最新モデルパス override')
	parser.add_argument('--config-json', type=str, default=None, help='設定上書き JSON')
	parser.add_argument('--batch-size', type=int, default=None, help='学習バッチサイズ override')
	parser.add_argument('--updates', type=int, default=500, help='今回実行する train_step 回数')
	parser.add_argument('--max-files', type=int, default=None, help='一度の取り込みで処理する新規 selfplay ファイル上限')
	parser.add_argument('--max-samples-per-file', type=int, default=None, help='各ファイルの取り込みサンプル上限 (train/val それぞれ)')
	parser.add_argument('--version-interval', type=int, default=10000, help='このエピソード累計間隔ごとに世代タグ付き ckpt 保存 (0=無効)')
	parser.add_argument('--device', type=str, default=None, help='デバイス指定 (auto/cpu/cuda)')
	parser.add_argument('--seed', type=int, default=None, help='乱数シード override')
	parser.add_argument('--hand-warmup-updates', type=int, default=None, help='開始から何更新まで hand_pred_loss_coef を一時増加させる (override config)')
	parser.add_argument('--hand-warmup-coef', type=float, default=None, help='ウォームアップ期間中に使用する一時的な hand_pred_loss_coef')
	args = parser.parse_args()

	base_cfg = _load_config(ALPHA_ZERO_CONFIG, args.config_json)
	if args.batch_size is not None:
		base_cfg['batch_size'] = int(args.batch_size)
	if args.device is not None:
		base_cfg['device'] = args.device
	if args.seed is not None:
		base_cfg['seed'] = int(args.seed)
	if args.model_path is not None:
		base_cfg['checkpoint_path'] = args.model_path
	# optional hand-warmup overrides (applied via config for train_loop)
	if args.hand_warmup_updates is not None:
		base_cfg['hand_warmup_updates'] = int(args.hand_warmup_updates)
	if args.hand_warmup_coef is not None:
		base_cfg['hand_warmup_coef'] = float(args.hand_warmup_coef)
	base_cfg['checkpoint_dir'] = args.checkpoint_dir
	base_cfg['log_dir'] = args.log_dir
	base_cfg['clear_logs_on_start'] = False  # 継続
	base_cfg['device'] = _resolve_device(base_cfg.get('device'))
	# 検証頻度（既定: 200 更新ごとに検証）
	base_cfg.setdefault('val_eval_every_updates', 200)
	# 検証は val 分割全体で評価（ばらつきを抑え、毎回の値が反映される）
	base_cfg.setdefault('val_use_full_split', True)
	# アクティブファイルプール刷新時に差分モードを既定で有効化
	base_cfg.setdefault('active_file_incremental_refresh', True)
	# DataLoader ワーカーの pickle 問題対策（Windows 非並列トレーナーでは既定で 0）
	base_cfg.setdefault('dataloader_num_workers', 0)
	# 起動時に限定的に読み込むファイル数（resume_defer_preload 有効時の初期ロード件数）
	base_cfg.setdefault('resume_initial_load_files', 20)

	train_loop(
		base_cfg,
		data_dir=args.data_dir,
		log_dir=args.log_dir,
		max_files=args.max_files,
		max_samples_per_file=args.max_samples_per_file,
		updates=int(args.updates),
		version_interval=int(args.version_interval),
	)


if __name__ == '__main__':
	# Windows multiprocessing 安定化
	try:
		import multiprocessing as _mp
		_mp.freeze_support()
	except Exception:
		pass
	main()

