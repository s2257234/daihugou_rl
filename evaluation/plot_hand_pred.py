"""手札予測精度のグラフ描画モジュール

残り手札枚数ごとの予測精度を可視化します。
"""
from __future__ import annotations

import os
import csv
from typing import Optional, Dict, List
import numpy as np


def plot_hand_prediction_accuracy(csv_path: str, output_dir: str) -> Optional[str]:
    """
    手札予測評価CSVからグラフを生成
    
    Args:
        csv_path: 手札予測評価CSVファイルのパス
        output_dir: 出力ディレクトリ
    
    Returns:
        生成されたグラフファイルのパス、またはNone
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # バックエンドを設定（GUI不要）
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plot generation")
        return None
    
    try:
        # CSVを読み込み
        hand_size_data: Dict[int, Dict[str, List[float]]] = {}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    hand_size = int(row.get('opponent_hand_size', 0))
                    if hand_size <= 0:
                        continue
                    
                    ai_acc = float(row.get('ai_accuracy', 0.0))
                    baseline_acc = float(row.get('baseline_accuracy', 0.0))
                    
                    if hand_size not in hand_size_data:
                        hand_size_data[hand_size] = {'ai': [], 'baseline': []}
                    
                    hand_size_data[hand_size]['ai'].append(ai_acc)
                    hand_size_data[hand_size]['baseline'].append(baseline_acc)
                except (ValueError, KeyError):
                    continue
        
        if not hand_size_data:
            print("[WARN] No hand prediction data found in CSV")
            return None
        
        # 残り手札枚数ごとの平均精度を計算
        hand_sizes = sorted(hand_size_data.keys(), reverse=True)  # 13→1の順
        ai_accuracies = []
        baseline_accuracies = []
        counts = []
        
        for hand_size in hand_sizes:
            ai_accs = hand_size_data[hand_size]['ai']
            baseline_accs = hand_size_data[hand_size]['baseline']
            
            if ai_accs and baseline_accs:
                ai_accuracies.append(np.mean(ai_accs))
                baseline_accuracies.append(np.mean(baseline_accs))
                counts.append(len(ai_accs))
            else:
                # データがない場合はスキップ
                continue
        
        if not ai_accuracies:
            print("[WARN] No valid accuracy data to plot")
            return None
        
        # グラフを描画
        plt.figure(figsize=(10, 6))
        
        # 青線: AIの精度
        plt.plot(hand_sizes[:len(ai_accuracies)], ai_accuracies, 
                'b-', linewidth=2, label='AI Accuracy', marker='o', markersize=6)
        
        # 点線: ベースライン（ランダム）の精度
        plt.plot(hand_sizes[:len(baseline_accuracies)], baseline_accuracies,
                'r--', linewidth=2, label='Baseline (Random)', marker='s', markersize=6)
        
        # グラフの設定
        plt.xlabel('Opponent Hand Size (Remaining Cards)', fontsize=12)
        plt.ylabel('Prediction Accuracy', fontsize=12)
        plt.title('Hand Prediction Accuracy by Remaining Hand Size', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0.0, 1.0)
        
        # X軸を13→1の逆順に設定
        plt.xticks(hand_sizes[:len(ai_accuracies)], hand_sizes[:len(ai_accuracies)])
        plt.gca().invert_xaxis()  # X軸を反転（13が左、1が右）
        
        # 各データポイントにサンプル数を表示（オプション）
        for i, (hs, cnt) in enumerate(zip(hand_sizes[:len(counts)], counts)):
            if i < len(ai_accuracies):
                plt.text(hs, ai_accuracies[i] + 0.02, f'n={cnt}', 
                        ha='center', va='bottom', fontsize=8, alpha=0.7)
        
        # 保存
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "hand_prediction_accuracy_by_hand_size.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"[WARN] Failed to plot hand prediction accuracy: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # テスト用
    import sys
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else "logs"
        result = plot_hand_prediction_accuracy(csv_path, output_dir)
        if result:
            print(f"Plot saved to: {result}")
        else:
            print("Failed to generate plot")
    else:
        print("Usage: python plot_hand_pred.py <csv_path> [output_dir]")
