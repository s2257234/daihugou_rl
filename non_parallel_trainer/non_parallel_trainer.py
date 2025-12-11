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
from typing import Any, Dict, List
import warnings

import joblib

# Ensure project root is on sys.path when run as a script
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJ_ROOT not in sys.path:
	sys.path.insert(0, _PROJ_ROOT)

from agents.config import ALPHA_ZERO_CONFIG
from agents.factory import create_env_and_agents
from agents.drl_agent import AlphaZeroAgent
from agents.replay_buffer import ReplayBuffer
from utils.logger import TrainingLogger

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
	"""Encapsulate selection logic (legacy ingest vs active pool)."""
	if files_override:
		return list(files_override)
	pool_size = int(cfg.get('active_file_pool_size', 0) or 0)
	newest_bias = float(cfg.get('active_file_newest_bias', 0.0) or 0.0)
	if pool_size > 0 and all_files:
		bias_count = int(min(pool_size, max(0, int(pool_size * newest_bias))))
		latest_slice = all_files[-bias_count:] if bias_count > 0 else []
		remaining_needed = pool_size - len(latest_slice)
		older_pool = all_files[:-bias_count] if bias_count > 0 else all_files
		if remaining_needed > 0 and older_pool:
			import random as _r
			active_rest = _r.sample(older_pool, min(remaining_needed, len(older_pool)))
		else:
			active_rest = []
		return latest_slice + active_rest
	# legacy ingest strategy
	try:
		eff_max_files = int(max_files) if (max_files is not None and int(max_files) > 0) else int(cfg.get('ingest_max_files') or 0)
	except Exception:
		eff_max_files = 0
	pick_newest = bool(cfg.get('ingest_pick_newest', True))
	if eff_max_files and eff_max_files > 0:
		return list(all_files[-eff_max_files:] if pick_newest else all_files[:eff_max_files])
	return list(all_files)


def _load_samples_for_files(
	files: List[str], *,
	max_samples_per_file: int | None,
	seed: int | None,
	val_ratio: float,
	logger: TrainingLogger | None,
	cfg: Dict[str, Any],
	preloaded: Dict[str, Dict[str, List[Dict[str, Any]]]] | None = None,
):
	"""Load samples, shuffle, split, return (train_list, val_list, total_train, total_val).

	If ``preloaded`` is provided it should be a mapping from file path to the dict
	returned by ``_load_samples_from_file``; in that case the function will reuse
	those entries and avoid calling ``joblib.load`` again. When ``max_samples_per_file``
	is provided the reused parts will be truncated to that limit.
	"""
	import random as _r
	all_samples: List[Dict[str, Any]] = []
	# Parallelize per-file loading where possible. If preloaded mapping is provided,
	# reuse entries; otherwise use joblib.Parallel to call _load_samples_from_file
	# concurrently. We measure per-file load time and emit lightweight events when
	# a logger is available.
	try:
		import time as _time, os as _os
		from joblib import Parallel, delayed
		# number of workers configurable via cfg; default to cpu_count
		n_jobs = int(cfg.get('resume_load_workers', 10) or 10)
		if n_jobs <= 0:
			try:
				import multiprocessing as _mp
				n_jobs = max(1, _mp.cpu_count() - 1)
			except Exception:
				n_jobs = 1

		# prepare list of files to actually load
		to_load = [fp for fp in files if not (preloaded is not None and fp in preloaded)]

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
			# Run parallel load; fall back to serial on any Parallel error
			try:
				results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_load_with_timing)(fp) for fp in to_load)
			except Exception:
				# fallback serial
				results = [_load_with_timing(fp) for fp in to_load]
		# build mapping fp -> parts
		parts_map: Dict[str, tuple] = {}
		for fp, parts_local, load_t, sz, cnt in results:
			parts_map[fp] = (parts_local, load_t, sz, cnt)
		# include preloaded entries with zero load time
		if preloaded is not None:
			for fp in files:
				if fp in preloaded:
					parts_map[fp] = (preloaded[fp], 0.0, 0, (len(preloaded[fp].get('train', []) or []) + len(preloaded[fp].get('val', []) or [])))

		# iterate in original order and append samples
		for fp in files:
			entry = parts_map.get(fp, ({'train': [], 'val': []}, 0.0, 0, 0))
			parts = entry[0]
			load_t = entry[1]
			sz = entry[2]
			cnt = entry[3]
			# honor max_samples_per_file truncation
			try:
				tr = parts.get('train', []) or []
				vl = parts.get('val', []) or []
				if max_samples_per_file is not None and int(max_samples_per_file) > 0:
					m = int(max_samples_per_file)
					if len(tr) > m:
						tr = random.sample(tr, m)  # ランダムサンプリング
					if len(vl) > m:
						vl = random.sample(vl, m)  # ランダムサンプリング
			except Exception:
				tr = parts.get('train', []) or []
				vl = parts.get('val', []) or []
			try:
				all_samples.extend(tr)
			except Exception:
				pass
			try:
				all_samples.extend(vl)
			except Exception:
				pass
	except Exception:
		# If anything went wrong with parallel path, fallback to original serial loop
		all_samples = []
		for fp in files:
			parts = None
			try:
				if preloaded is not None and fp in preloaded:
					parts = preloaded[fp]
				else:
					parts = _load_samples_from_file(fp, max_samples=(max_samples_per_file if max_samples_per_file else None))
			except Exception:
				parts = {'train': [], 'val': []}

			# If caller requested a per-file max_samples, ensure we honor it even when
			# reusing preloaded content.
			try:
				tr = parts.get('train', []) or []
				vl = parts.get('val', []) or []
				if max_samples_per_file is not None and int(max_samples_per_file) > 0:
					m = int(max_samples_per_file)
					if len(tr) > m:
						tr = random.sample(tr, m)  # ランダムサンプリング
					if len(vl) > m:
						vl = random.sample(vl, m)  # ランダムサンプリング
			except Exception:
				tr = parts.get('train', []) or []
				vl = parts.get('val', []) or []

			try:
				all_samples.extend(tr)
			except Exception:
				pass
			try:
				all_samples.extend(vl)
			except Exception:
				pass

	try:
		if seed is not None:
			_r.Random(int(seed)).shuffle(all_samples)
		else:
			_r.shuffle(all_samples)
	except Exception:
		pass

	split_idx = int(len(all_samples) * (1.0 - val_ratio))
	train_part = all_samples[:split_idx]
	val_part = all_samples[split_idx:]
	return train_part, val_part, len(train_part), len(val_part)


def _rebuild_replay(shared_rb: ReplayBuffer, train_part: List[Dict[str, Any]], val_part: List[Dict[str, Any]]):
	"""Clear and repopulate replay buffer with provided parts."""
	try:
		shared_rb.clear()
	except Exception:
		pass
	train_total = 0
	val_total = 0
	# Prepare combined list to bulk-extend for lower overhead
	combined = []
	for s in train_part:
		if isinstance(s, dict):
			try:
				s['split'] = 'train'
				combined.append(s)
			except Exception:
				pass
	for s in val_part:
		if isinstance(s, dict):
			try:
				s['split'] = 'val'
				combined.append(s)
			except Exception:
				pass
	# Use ReplayBuffer.extend when available
	try:
		uids = []
		if hasattr(shared_rb, 'extend'):
			uids = shared_rb.extend(combined)
			# estimate counts from uids length proportional to train/val ordering
			train_total = sum(1 for s in combined if s.get('split') == 'train')
			val_total = sum(1 for s in combined if s.get('split') == 'val')
		else:
			for s in combined:
				try:
					shared_rb.append(s)
					if s.get('split') == 'train':
						train_total += 1
					else:
						val_total += 1
				except Exception:
					pass
	except Exception:
		# fallback to per-item append in case of unexpected errors
		train_total = 0
		val_total = 0
		for s in combined:
			if isinstance(s, dict):
				try:
					shared_rb.append(s)
					if s.get('split') == 'train':
						train_total += 1
					else:
						val_total += 1
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
		if not bool(cfg.get('enable_dynamic_sample_sizing', False)):
			return None
		
		exponent = float(cfg.get('dynamic_sample_exponent', 0.75))
		base = float(cfg.get('dynamic_sample_base', 1.0))
		min_per_file = int(cfg.get('dynamic_sample_min_per_file', 500))
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


def _refresh_active_pool(active_files: List[str], all_files: List[str], cfg: Dict[str, Any], shared_rb: ReplayBuffer, val_ratio: float, seed: int | None, max_samples_per_file: int | None, logger: TrainingLogger | None, train_updates_cum: int = 0):
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
		train_part, val_part, tcount, vcount = _load_samples_for_files(active_files_new, max_samples_per_file=max_samples_per_file, seed=seed, val_ratio=val_ratio, logger=logger, cfg=cfg)
		loaded_train, loaded_val = _rebuild_replay(shared_rb, train_part, val_part)
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
		for fp in new_add:
			parts = _load_samples_from_file(fp, max_samples=(int(max_samples_per_file) if (max_samples_per_file is not None and max_samples_per_file > 0) else None))
			tr = parts.get('train', []) or []
			vl = parts.get('val', []) or []
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
	for s in samples:
		if not isinstance(s, dict):
			continue
		sp = s.get('split', 'train')
		# ソースファイルを保持し、後段の差分更新や分析を可能にする
		try:
			s['source_file'] = path
		except Exception:
			pass
		if sp == 'val':
			val_list.append(s)
		else:
			train_list.append(s)
	if max_samples is not None and max_samples > 0:
		if len(train_list) > max_samples:
			# ランダムサンプリングで多様性を確保（過学習対策）
			train_list = random.sample(train_list, max_samples)
		if len(val_list) > max_samples:
			val_list = random.sample(val_list, max_samples)
	return {'train': train_list, 'val': val_list}


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
	try:
		vinfo = learner.validate_step(batch_size=bs)
	except Exception:
		vinfo = {"policy_loss": None, "value_loss": None, "hand_pred_loss": None}
	if isinstance(vinfo, dict) and (vinfo.get("policy_loss") is not None or vinfo.get("value_loss") is not None or vinfo.get("hand_pred_loss") is not None):
		logger.log_validation(vinfo)
		
	else:
		logger.log_text("[val] no validation metrics (val split may be empty)")
		


def _save_checkpoint(bundle, cfg: Dict[str, Any], model_version: int, version_tag: str | None = None):
	os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
	model = bundle.model
	learner = bundle.agents[cfg.get('learning_player_id', 0)]
	latest_path = cfg.get('checkpoint_path', os.path.join(cfg['checkpoint_dir'], 'policy_value_latest.pt'))
	def _atomic(pt: str):
		tmp = pt + '.tmp'
		try:
			model.save(tmp, force_sync=True)  # type: ignore[arg-type]
			os.replace(tmp, pt)
		except Exception:
			try:
				if os.path.exists(tmp):
					os.remove(tmp)
			except Exception:
				pass
			model.save(pt)  # type: ignore[arg-type]
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

	# モデル/エージェント/環境生成 (今回は永続リプレイを使わずエフェメラル)
	shared_rb = ReplayBuffer(maxlen=cfg.get('buffer_size', 50000), path=None)
	# 直列モードではリプレイ縮小サイクルを無効化（設定・ロガー登録もしない）

	bundle = create_env_and_agents(
		cfg,
		context='main',
		model_path=cfg.get('checkpoint_path'),
		shared_replay=shared_rb,
		logger=logger,
	)

	learner: AlphaZeroAgent = bundle.agents[cfg.get('learning_player_id', 0)]

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
		if os.path.isfile(opt_p):
			if hasattr(learner, 'ensure_optimizer'):
				learner.ensure_optimizer()
			ok = False
			if hasattr(learner, 'load_optimizer'):
				ok = bool(learner.load_optimizer(opt_p, map_location=bundle.device))
			logger.log_text(f"[resume] optimizer load {'ok' if ok else 'failed'} from {opt_p}")
	except Exception as e:
		logger.log_text(f"[WARN] optimizer load failed: {e}")
	try:
		sch_p = os.path.join(cfg['checkpoint_dir'], 'scheduler_latest.pt')
		if os.path.isfile(sch_p):
			if hasattr(learner, 'ensure_scheduler'):
				learner.ensure_scheduler()
			ok = False
			if hasattr(learner, 'load_scheduler'):
				ok = bool(learner.load_scheduler(sch_p))
			logger.log_text(f"[resume] scheduler load {'ok' if ok else 'failed'} from {sch_p}")
	except Exception as e:
		logger.log_text(f"[WARN] scheduler load failed: {e}")

	# meta.json + 既存CSV から累積アップデート数を復元（CSV優先）
	meta = _load_meta(data_dir)
	meta_updates = int(meta.get('train_updates_cumulative', 0) or 0)
	episodes_cumulative = int(meta.get('episodes_cumulative', 0) or 0)
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
	

	# --- Active File Pool (modular) ---
	all_files, pattern_attempts = _scan_selfplay_files(data_dir)
	found_count = len(all_files)
	pool_size = int(cfg.get('active_file_pool_size', 0) or 0)
	refresh_every = int(cfg.get('active_file_refresh_every_updates', 0) or 0)
	refresh_fraction = float(cfg.get('active_file_pool_refresh_fraction', 0.0) or 0.0)
	active_files = _select_initial_active_files(all_files, cfg, max_files=max_files, files_override=files_override)
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
	initial_active_files = list(active_files) if active_files is not None else []
	try:
		# Optionally defer preloading heavy sample files after resume.
		# If `resume_defer_preload` is True in config, skip loading files here to avoid long blocking I/O.
		if bool(cfg.get('resume_defer_preload', True)):
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
		dynamic_result = _compute_dynamic_file_and_sample_params(cfg, train_updates_cum, len(initial_active_files), logger)
		if dynamic_result is not None:
			_, eff_max_samp = dynamic_result  # サンプル数のみ使用（ファイル数は既に適用済み）
		else:
			eff_max_samp = max_samples_per_file if (max_samples_per_file is not None and max_samples_per_file > 0) else (cfg.get('ingest_max_samples_per_file') or None)
		if eff_max_samp: eff_max_samp = int(eff_max_samp)
	except Exception:
		eff_max_samp = None
	# Use initial_active_files for the first rebuild to limit startup I/O when resume_defer_preload is set.
	# Instrumentation: measure durations of sample loading and replay rebuild to diagnose long resume pauses.
	try:
		resume_timer_start = time.time()
		if logger:
			logger.log_text("[resume-timer] starting resume preload")
			logger.flush_buffers(force=True)
	except Exception:
		pass
	# load samples with timing
	t_load_start = time.time()
	if logger:
		try:
			logger.log_text(f"[resume-timer] before_load_samples files={len(initial_active_files)}")
			logger.flush_buffers(force=True)
		except Exception:
			pass
	train_part, val_part, train_samples_total, val_samples_total = _load_samples_for_files(initial_active_files, max_samples_per_file=eff_max_samp, seed=_seed, val_ratio=val_ratio, logger=logger, cfg=cfg, preloaded=preloaded)
	t_load_end = time.time()
	try:
		if logger:
			logger.log_text(f"[resume-timer] load_samples took={t_load_end - t_load_start:.3f}s files={len(initial_active_files)} samples_train={train_samples_total} samples_val={val_samples_total}")
			logger.flush_buffers(force=True)
	except Exception:
		pass
	# rebuild replay with timing
	t_rebuild_start = time.time()
	train_samples_total, val_samples_total = _rebuild_replay(shared_rb, train_part, val_part)
	t_rebuild_end = time.time()
	try:
		if logger:
			logger.log_text(f"[resume-timer] rebuild_replay took={t_rebuild_end - t_rebuild_start:.3f}s appended_train={train_samples_total} appended_val={val_samples_total}")
			logger.flush_buffers(force=True)
	except Exception:
		pass
	# 集約進捗表示 (episodes は meta から取得した累積値)
	# replay_size は今回選択された全ファイル（max_filesで切り捨て前）の合計サンプル数を表示する
	# 集約進捗表示 (episodes は meta から取得した累積値)
	# replay_size は実際のリプレイバッファ長 (上限25万・到達時は10万へ削減) を表示する
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
	updates = int(updates)
	val_every = int(cfg.get('val_eval_every_updates', 0) or 0)
	batch_size = int(cfg.get('batch_size', 256) or 256)
	version_interval = int(version_interval or 0)
	model_version = int(meta.get('model_version', 0) or 0)  # self-play と共有する番号を継続利用
	# バリデーションをログ出力と同一タイミングに揃える: ログ間隔を val_every (設定されていれば) に合わせる
	log_interval = val_every if val_every > 0 else int(cfg.get('train_log_every_updates', 1) or 1)

	last_loss_info = None
	# --- 学習ループ + プールリフレッシュ ---
	for i in range(updates):
		loss_info = learner.train_step(batch_size=batch_size)
		# 空データ回避ログ
		if isinstance(loss_info, dict) and loss_info.get('loss') is None and loss_info.get('reason') == 'no_data':
			logger.log_text('[warn] train_step skipped (no_data)')
		# アクティブファイルプール刷新（更新回数ベース）
		if pool_size > 0 and refresh_every > 0 and ((train_updates_cum + i + 1) % refresh_every == 0):
			try:
				active_files, _stats = _refresh_active_pool(active_files, all_files, cfg, shared_rb, val_ratio, _seed, eff_max_samp, logger, train_updates_cum=train_updates_cum + i + 1)
			except Exception as e:
				logger.log_text(f"[WARN] active-pool refresh failed: {e}")

		# ログ間隔に到達したときのみ検証 & ログ出力
		if ((i + 1) % log_interval) == 0:
			if val_every > 0:  # バリデーション間隔設定がある場合はそのタイミングで実施
				_run_validation(learner, logger, cfg, batch_size=batch_size, note=f'update={train_updates_cum + i + 1}')
			if isinstance(loss_info, dict) and loss_info.get('loss') is not None:
				try:
					loss_info.setdefault('train_count', train_updates_cum + i + 1)
				except Exception:
					pass
				logger.log_train(loss_info)
				last_loss_info = loss_info

	# 最後に最新モデル保存 (version_tag なし)
	_save_checkpoint(bundle, cfg, model_version, version_tag=None)

	# エピソード累計に基づくバージョン保存: 指定間隔ごとに ep タグで保存
	try:
		if version_interval > 0 and episodes_cumulative > 0 and (episodes_cumulative % version_interval == 0):
			model_version += 1
			_save_checkpoint(bundle, cfg, model_version, version_tag=f'ep{episodes_cumulative}')
			logger.log_text(f"[ckpt] saved version={model_version} at episode={episodes_cumulative}")
	except Exception:
		pass

	# 学習後の最終検証を一度実施（intervalに依らず終端の値を残す）
	_run_validation(learner, logger, cfg, batch_size=batch_size, note='post-train')

	train_updates_cum += updates
	# consumed_files は使用しない (毎回全ファイルを対象)
	meta['train_updates_cumulative'] = train_updates_cum
	meta['model_version'] = model_version
	meta['episodes_cumulative'] = episodes_cumulative  # 既存値保持 (self-play 側で更新済み想定)
	# リプレイサイズを meta に記録しておくと self-play 側から軽量に参照できる
	# meta へも全選択ファイル合計サンプル数を記録（学習に使った一部だけでなく全体規模を把握する目的）
	# リプレイサイズを meta に記録（表示/自己対局側参照用）。実メモリ内のバッファ長を採用。
	try:
		meta['replay_size'] = int(len(shared_rb))
	except Exception:
		pass
	logger.log_text(f"[summary-train] updates={updates} total_updates={train_updates_cum} model_version={model_version} replay_size={_total_samples_all}")
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
	logger.log_text(f"[summary-train] updates={updates} total_updates={train_updates_cum} model_version={model_version} replay_size={_total_samples_all}")
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
	parser.add_argument('--version-interval', type=int, default=0, help='このエピソード累計間隔ごとに世代タグ付き ckpt 保存 (0=無効)')
	parser.add_argument('--device', type=str, default=None, help='デバイス指定 (auto/cpu/cuda)')
	parser.add_argument('--seed', type=int, default=None, help='乱数シード override')
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

