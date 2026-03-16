"""
訓練ログからValue/Policyのトレーニング・検証ロスをプロットするスクリプト

使用方法:
    python tools/plot_losses.py
    python tools/plot_losses.py --csv logs/train_updates.csv --output losses.png
    python tools/plot_losses.py --show  # 画像を表示（保存もする）
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib
# Use non-interactive backend by default to avoid Tkinter-related errors
# (e.g. "PyCapsule_New called with null pointer" when Tk isn't available)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import glob
import numpy as np

# プロジェクトルートをパスに追加
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def load_data(csv_path: str) -> pd.DataFrame:
    """CSVファイルを読み込む"""
    df = pd.read_csv(csv_path)
    # 数値列のみを使用
    numeric_cols = ['update_step', 'policy_loss', 'value_loss', 
                    'val_policy_loss', 'val_value_loss', 'hand_pred_loss', 'val_hand_pred_loss',
                    # ValueTarget columns possibly emitted into train_updates.csv
                    'vt_count','vt_mean','vt_std','vt_min','vt_max','vt_clip_neg','vt_clip_pos']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def plot_losses(df: pd.DataFrame, output_path: str, show: bool = False):
    """ロスをプロットする"""
    # 有効なデータのみをフィルタ
    df = df.dropna(subset=['update_step'])
    
    # 1x3のサブプロットを作成（横一列）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training and Validation Losses', fontsize=14, fontweight='bold')
    
    # 1. Policy Loss
    ax1 = axes[0]
    if 'policy_loss' in df.columns:
        valid = df['policy_loss'].notna()
        ax1.plot(df.loc[valid, 'update_step'], df.loc[valid, 'policy_loss'], 
             label='Train Policy Loss', color='blue', alpha=0.8, linewidth=1)
    if 'val_policy_loss' in df.columns:
        valid = df['val_policy_loss'].notna()
        ax1.plot(df.loc[valid, 'update_step'], df.loc[valid, 'val_policy_loss'], 
             label='Val Policy Loss', color='red', alpha=0.8, linewidth=1)
    ax1.set_xlabel('Update Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Policy Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Value Loss
    ax2 = axes[1]
    if 'value_loss' in df.columns:
        valid = df['value_loss'].notna()
        ax2.plot(df.loc[valid, 'update_step'], df.loc[valid, 'value_loss'], 
             label='Train Value Loss', color='blue', alpha=0.8, linewidth=1)
    if 'val_value_loss' in df.columns:
        valid = df['val_value_loss'].notna()
        ax2.plot(df.loc[valid, 'update_step'], df.loc[valid, 'val_value_loss'], 
             label='Val Value Loss', color='red', alpha=0.8, linewidth=1)
    ax2.set_xlabel('Update Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Value Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Hand Prediction Loss
    ax3 = axes[2]
    if 'hand_pred_loss' in df.columns:
        valid = df['hand_pred_loss'].notna()
        ax3.plot(df.loc[valid, 'update_step'], df.loc[valid, 'hand_pred_loss'], 
             label='Train Hand Loss', color='blue', alpha=0.8, linewidth=1)
    if 'val_hand_pred_loss' in df.columns:
        valid = df['val_hand_pred_loss'].notna()
        ax3.plot(df.loc[valid, 'update_step'], df.loc[valid, 'val_hand_pred_loss'], 
             label='Val Hand Loss', color='red', alpha=0.8, linewidth=1)
    ax3.set_xlabel('Update Step')
    ax3.set_ylabel('Loss')
    ax3.set_title('Hand Prediction Loss')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

    # 別図: ValueTarget の平均 / std をプロット
    try:
        if 'vt_mean' in df.columns:
            valid = df['vt_mean'].notna()
            if valid.any():
                fig2, ax = plt.subplots(1, 1, figsize=(10, 4))
                ax.plot(df.loc[valid, 'update_step'], df.loc[valid, 'vt_mean'], label='ValueTarget Mean', color='tab:orange')
                if 'vt_std' in df.columns:
                    std_valid = df['vt_std'].notna()
                    if std_valid.any():
                        # 合わせるために共通の有効インデックスを採用
                        use = valid & std_valid
                        if use.any():
                            x = df.loc[use, 'update_step']
                            m = df.loc[use, 'vt_mean']
                            s = df.loc[use, 'vt_std']
                            ax.fill_between(x, m - s, m + s, color='tab:orange', alpha=0.2, label='±std')
                ax.set_xlabel('Update Step')
                ax.set_ylabel('ValueTarget Mean')
                ax.set_title('ValueTarget Mean over Updates')
                ax.grid(True, alpha=0.3)
                ax.legend()
                vt_out = os.path.join(os.path.dirname(output_path), 'value_targets_plot.png')
                fig2.tight_layout()
                fig2.savefig(vt_out, dpi=150, bbox_inches='tight')
                print(f"[INFO] Saved ValueTarget plot to {vt_out}")
                if show:
                    fig2.show()
                else:
                    plt.close(fig2)
    except Exception as e:
        print(f"[WARN] Failed to produce ValueTarget plot: {e}")


def plot_summary_stats(df: pd.DataFrame):
    """統計情報を表示"""
    print("\n" + "="*60)
    print("Training Summary Statistics")
    print("="*60)
    
    # 最初と最後のステップ
    first_step = df['update_step'].min()
    last_step = df['update_step'].max()
    print(f"Update Steps: {first_step:.0f} → {last_step:.0f}")
    
    # 各ロスの推移
    for loss_name, train_col, val_col in [
        ('Policy', 'policy_loss', 'val_policy_loss'),
        ('Value', 'value_loss', 'val_value_loss'),
        ('Hand Pred', 'hand_pred_loss', 'val_hand_pred_loss'),
    ]:
        if train_col in df.columns:
            valid = df[train_col].notna()
            if valid.any():
                first_val = df.loc[valid, train_col].iloc[0]
                last_val = df.loc[valid, train_col].iloc[-1]
                min_val = df.loc[valid, train_col].min()
                print(f"\n{loss_name} Loss (Train):")
                print(f"  First: {first_val:.4f} → Last: {last_val:.4f} (Min: {min_val:.4f})")
        
        if val_col in df.columns:
            valid = df[val_col].notna()
            if valid.any():
                first_val = df.loc[valid, val_col].iloc[0]
                last_val = df.loc[valid, val_col].iloc[-1]
                min_val = df.loc[valid, val_col].min()
                print(f"{loss_name} Loss (Val):")
                print(f"  First: {first_val:.4f} → Last: {last_val:.4f} (Min: {min_val:.4f})")
    
    # 過学習判定
    if 'value_loss' in df.columns and 'val_value_loss' in df.columns:
        valid = df['value_loss'].notna() & df['val_value_loss'].notna()
        if valid.any():
            last_train = df.loc[valid, 'value_loss'].iloc[-1]
            last_val = df.loc[valid, 'val_value_loss'].iloc[-1]
            gap = last_val - last_train
            print(f"\n[Overfitting Check] Value Loss Gap (Val - Train): {gap:.4f}")
            if gap > 0.1:
                print("  ⚠️ Warning: Possible overfitting detected!")
            elif gap > 0.05:
                print("  ⚡ Mild overfitting trend")
            else:
                print("  ✅ No significant overfitting")
    
    print("="*60)

    # --- Value target distribution diagnostics (from latest selfplay joblib if available) ---
    try:
        data_dir = os.path.join(_PROJ_ROOT, 'data')
        pattern = os.path.join(data_dir, 'selfplay_ep*.joblib')
        files = glob.glob(pattern)
        latest = None
        if files:
            latest = max(files, key=lambda p: os.path.getmtime(p))
        if latest is None:
            print('[ValueTarget] No selfplay joblib found in data/ to compute value-target distribution')
            return

        print(f"[ValueTarget] Loading latest selfplay joblib: {os.path.basename(latest)}")
        obj = joblib.load(latest)

        samples = []
        if isinstance(obj, dict):
            for k in ('samples', 'replay', 'data', 'episodes'):
                if k in obj and isinstance(obj[k], (list, tuple)):
                    samples = list(obj[k]); break
            if not samples and ('pi_q' in obj or 'value' in obj):
                samples = [obj]
        elif isinstance(obj, (list, tuple)):
            samples = list(obj)

        vals = []
        vals_train = []
        vals_val = []
        for s in samples:
            try:
                v = None
                if isinstance(s, dict):
                    v = s.get('value')
                else:
                    continue
                if v is None:
                    continue
                v = float(v)
                vals.append(v)
                if isinstance(s, dict) and 'split' in s and s.get('split') == 'val':
                    vals_val.append(v)
                else:
                    vals_train.append(v)
            except Exception:
                continue

        if len(vals) == 0:
            print('[ValueTarget] No value targets found in latest selfplay samples')
            return

        def _print_stats(name, arr):
            a = np.asarray(arr, dtype=np.float64)
            print(f'-- {name} count: {a.size}')
            if a.size > 0:
                print(f'   mean: {a.mean():.4f}, std: {a.std():.4f}, min: {a.min():.4f}, max: {a.max():.4f}')
                # histogram -1..+1
                bins = np.linspace(-1.0, 1.0, 41)
                h, edges = np.histogram(a, bins=bins)
                # print compact histogram summary
                print('   histogram bins(-1..+1, 40):', h.tolist())
                # check clamping
                low_clip = float((a <= -0.9999).sum())
                high_clip = float((a >= 0.9999).sum())
                print(f'   clipped at -1: {int(low_clip)} ({low_clip/a.size:.3f}), +1: {int(high_clip)} ({high_clip/a.size:.3f})')

        print('\n[ValueTarget] Overall distribution (from latest selfplay samples)')
        _print_stats('ALL', vals)
        if vals_train:
            _print_stats('TRAIN (default split)', vals_train)
        if vals_val:
            _print_stats('VAL (split=="val")', vals_val)

            # Note on time evolution: to check epoch-wise shrinkage you need historical joblibs per epoch.
            print('\n[ValueTarget] Note: For epoch-wise shrinkage check, run analysis across historical selfplay files or')
            print('  record value-target stats per training epoch when generating selfplay data.')
    except Exception as e:
        print(f'[ValueTarget] Error while computing value-target distribution: {e}')

    # --- ValueTarget stats from CSV if present ---
    try:
        if 'vt_mean' in df.columns:
            valid = df['vt_mean'].notna()
            if valid.any():
                first_v = df.loc[valid, 'vt_mean'].iloc[0]
                last_v = df.loc[valid, 'vt_mean'].iloc[-1]
                min_v = df.loc[valid, 'vt_mean'].min()
                max_v = df.loc[valid, 'vt_mean'].max()
                print('\n[ValueTarget][CSV] vt_mean:')
                print(f'  First: {first_v:.4f} → Last: {last_v:.4f} (Min: {min_v:.4f}, Max: {max_v:.4f})')
                if 'vt_count' in df.columns:
                    print(f'  Latest count: {int(df.loc[valid, "vt_count"].iloc[-1]) if not np.isnan(df.loc[valid, "vt_count"].iloc[-1]) else "N/A"}')
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description='Plot training and validation losses')
    parser.add_argument('--csv', type=str, default='logs/train_updates.csv',
                        help='Path to train_updates.csv')
    parser.add_argument('--output', type=str, default='logs/losses_plot.png',
                        help='Output path for the plot image')
    parser.add_argument('--show', action='store_true',
                        help='Show the plot in a window')
    parser.add_argument('--stats', action='store_true',
                        help='Print summary statistics')
    args = parser.parse_args()
    
    # パスを調整
    if not os.path.isabs(args.csv):
        args.csv = os.path.join(_PROJ_ROOT, args.csv)
    if not os.path.isabs(args.output):
        args.output = os.path.join(_PROJ_ROOT, args.output)
    
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV file not found: {args.csv}")
        sys.exit(1)
    
    print(f"[INFO] Loading data from {args.csv}")
    df = load_data(args.csv)
    print(f"[INFO] Loaded {len(df)} rows")
    
    if args.stats or True:  # 常に統計を表示
        plot_summary_stats(df)
    
    plot_losses(df, args.output, show=args.show)


if __name__ == '__main__':
    main()
