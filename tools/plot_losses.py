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
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def load_data(csv_path: str) -> pd.DataFrame:
    """CSVファイルを読み込む"""
    df = pd.read_csv(csv_path)
    # 数値列のみを使用
    numeric_cols = ['update_step', 'policy_loss', 'value_loss', 
                    'val_policy_loss', 'val_value_loss', 'hand_pred_loss', 'val_hand_pred_loss']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def plot_losses(df: pd.DataFrame, output_path: str, show: bool = False):
    """ロスをプロットする"""
    # 有効なデータのみをフィルタ
    df = df.dropna(subset=['update_step'])
    
    # 2x2のサブプロットを作成
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training and Validation Losses', fontsize=14, fontweight='bold')
    
    # 1. Policy Loss
    ax1 = axes[0, 0]
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
    ax2 = axes[0, 1]
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
    ax3 = axes[1, 0]
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
    
    # 4. Train/Val Gap (過学習の指標)
    ax4 = axes[1, 1]
    if 'value_loss' in df.columns and 'val_value_loss' in df.columns:
        valid = df['value_loss'].notna() & df['val_value_loss'].notna()
        gap = df.loc[valid, 'val_value_loss'] - df.loc[valid, 'value_loss']
        ax4.plot(df.loc[valid, 'update_step'], gap, 
                 label='Value Loss Gap (Val - Train)', color='purple', alpha=0.8, linewidth=1)
    if 'policy_loss' in df.columns and 'val_policy_loss' in df.columns:
        valid = df['policy_loss'].notna() & df['val_policy_loss'].notna()
        gap = df.loc[valid, 'val_policy_loss'] - df.loc[valid, 'policy_loss']
        ax4.plot(df.loc[valid, 'update_step'], gap, 
                 label='Policy Loss Gap (Val - Train)', color='green', alpha=0.8, linewidth=1)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.5)
    ax4.set_xlabel('Update Step')
    ax4.set_ylabel('Gap (Val - Train)')
    ax4.set_title('Overfitting Indicator (Gap > 0 = Overfitting)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


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
