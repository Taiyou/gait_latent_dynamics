#!/usr/bin/env python3
"""
Demixed PCA Demo: Gait Dynamics × Mental State Analysis

歩行ダイナミクスとメンタル状態の対応関係を分析するデモスクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt

from src.dpca import DemixedPCA, GaitDPCA
from src.data_loader import generate_synthetic_gait_data, GaitDataLoader
from src.visualization import DPCAVisualizer, quick_plot_dpca_results


def main():
    """メインのデモ実行関数"""
    
    print("=" * 60)
    print("Demixed PCA Analysis: Gait Dynamics × Mental State")
    print("=" * 60)
    
    # ==========================================================
    # 1. データ生成
    # ==========================================================
    print("\n[1] Generating synthetic gait data...")
    
    # 合成歩行データを生成
    # - 50トライアル
    # - 15個の歩行特徴量（関節角度、GRF等）
    # - 100時間点（歩行周期0-100%）
    # - 5つのメンタル状態条件
    data, metadata = generate_synthetic_gait_data(
        n_trials=50,
        n_features=15,
        n_timepoints=100,
        n_conditions=5,
        mental_state_effect_strength=0.3,
        time_effect_strength=0.5,
        interaction_strength=0.15,
        noise_level=0.1,
        random_state=42
    )
    
    print(f"   Data shape: {data.shape}")
    print(f"   (n_trials, n_features, n_timepoints, n_conditions)")
    print(f"   Features: {metadata['feature_labels']}")
    print(f"   Mental states: {metadata['mental_state_labels']}")
    
    # ==========================================================
    # 2. データプリプロセッシング
    # ==========================================================
    print("\n[2] Preprocessing data...")
    
    loader = GaitDataLoader()
    
    # 正規化とフィルタリング
    data_processed = loader.preprocess(
        data,
        normalize=True,
        filter_cutoff=0.8,  # 80%のローパスフィルタ
        resample_points=None  # リサンプリングなし
    )
    
    # トライアル平均
    data_averaged = loader.get_trial_averaged_data(data_processed)
    print(f"   Averaged data shape: {data_averaged.shape}")
    
    # ==========================================================
    # 3. Demixed PCA フィッティング
    # ==========================================================
    print("\n[3] Fitting Demixed PCA model...")
    
    # dPCAモデルを作成
    dpca = DemixedPCA(
        n_components=10,
        regularizer='auto'
    )
    
    # フィット
    dpca.fit(
        data_averaged,
        feature_labels=metadata['feature_labels'],
        marginalization_labels=['time', 'condition', 'interaction']
    )
    
    print("   Model fitted successfully!")
    print(f"   Regularization parameter: {dpca._regularizer_value:.2e}")
    
    # ==========================================================
    # 4. 結果の分析
    # ==========================================================
    print("\n[4] Analyzing results...")
    
    # 分散分解のサマリー
    summary = dpca.get_demixing_summary()
    print("\n   Variance Decomposition:")
    for name, var in summary.items():
        print(f"      {name}: {var:.1%}")
    
    # 各marginalizationの説明分散
    print("\n   Top 3 components per marginalization:")
    for margin_name in dpca.explained_variance_ratio_:
        var_ratios = dpca.explained_variance_ratio_[margin_name][:3]
        print(f"      {margin_name}: {[f'{v:.1%}' for v in var_ratios]}")
    
    # ==========================================================
    # 5. データ変換
    # ==========================================================
    print("\n[5] Transforming data...")
    
    # 全marginalizationで変換
    transformed = dpca.transform(data_averaged)
    
    for name, trans in transformed.items():
        print(f"   {name}: {trans.shape}")
    
    # 特定のmarginalzationのみ
    time_components = dpca.transform(data_averaged, marginalization='time')
    print(f"\n   Time components shape: {time_components.shape}")
    
    # ==========================================================
    # 6. 可視化
    # ==========================================================
    print("\n[6] Creating visualizations...")
    
    viz = DPCAVisualizer()
    
    # 出力ディレクトリの作成
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # 6.1 包括的サマリー
    fig_summary = viz.plot_comprehensive_summary(
        dpca,
        data_averaged,
        metadata['mental_state_labels'],
        metadata['feature_labels'],
        save_path=str(output_dir / 'dpca_summary.png')
    )
    print(f"   Saved: {output_dir / 'dpca_summary.png'}")
    
    # 6.2 説明分散
    fig_var = viz.plot_explained_variance(
        dpca,
        save_path=str(output_dir / 'explained_variance.png')
    )
    print(f"   Saved: {output_dir / 'explained_variance.png'}")
    
    # 6.3 成分の時系列（条件=メンタル状態依存成分）
    fig_timecourse = viz.plot_component_timecourse(
        dpca,
        data_averaged,
        marginalization='condition',
        components=[0, 1, 2],
        mental_state_labels=metadata['mental_state_labels'],
        save_path=str(output_dir / 'condition_timecourse.png')
    )
    print(f"   Saved: {output_dir / 'condition_timecourse.png'}")
    
    # 6.4 メンタル状態の分離
    fig_separation = viz.plot_mental_state_separation(
        dpca,
        data_averaged,
        metadata['mental_state_labels'],
        marginalization='condition',
        save_path=str(output_dir / 'mental_state_separation.png')
    )
    print(f"   Saved: {output_dir / 'mental_state_separation.png'}")
    
    # 6.5 成分重み
    fig_weights = viz.plot_component_weights(
        dpca,
        marginalization='condition',
        component=0,
        feature_labels=metadata['feature_labels'],
        save_path=str(output_dir / 'component_weights.png')
    )
    print(f"   Saved: {output_dir / 'component_weights.png'}")
    
    # 6.6 再構成品質
    fig_recon = viz.plot_reconstruction_quality(
        dpca,
        data_averaged,
        feature_idx=0,
        condition_idx=0,
        save_path=str(output_dir / 'reconstruction_quality.png')
    )
    print(f"   Saved: {output_dir / 'reconstruction_quality.png'}")
    
    # ==========================================================
    # 7. メンタル状態の分離度分析（GaitDPCA）
    # ==========================================================
    print("\n[7] Mental state separation analysis (GaitDPCA)...")
    
    gait_dpca = GaitDPCA(n_components=10)
    gait_dpca.fit_with_labels(
        data_averaged,
        mental_state_labels=metadata['mental_state_labels'],
        feature_labels=metadata['feature_labels']
    )
    
    separation_scores = gait_dpca.analyze_mental_state_separation(
        data_averaged,
        metadata['mental_state_labels']
    )
    
    print("\n   Mental State Separation Scores:")
    sorted_scores = sorted(separation_scores.items(), key=lambda x: x[1], reverse=True)
    for pair, score in sorted_scores:
        print(f"      {pair}: {score:.3f}")
    
    # ==========================================================
    # 8. 完了
    # ==========================================================
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 60)
    
    # プロットを表示
    plt.show()
    
    return dpca, data_averaged, metadata


def run_significance_analysis():
    """有意性検定のデモ"""
    
    print("\n" + "=" * 60)
    print("Running Significance Analysis...")
    print("=" * 60)
    
    # データ生成（トライアルレベルデータが必要）
    data, metadata = generate_synthetic_gait_data(
        n_trials=30,
        n_features=10,
        n_timepoints=50,
        n_conditions=4,
        random_state=42
    )
    
    # プリプロセス
    loader = GaitDataLoader()
    data_processed = loader.preprocess(data, normalize=True)
    data_averaged = data_processed.mean(axis=0)
    
    # dPCAフィット
    dpca = DemixedPCA(n_components=5)
    dpca.fit(data_averaged)
    
    # 有意性検定（シャッフル数を減らしてデモ用に高速化）
    print("\nRunning shuffle test (this may take a moment)...")
    significance = dpca.significance_analysis(
        data_processed,
        n_shuffles=100,  # デモ用に少なめ
        alpha=0.05
    )
    
    print("\nSignificant components (p < 0.05):")
    for margin_name, sig_mask in significance.items():
        n_sig = sig_mask.sum()
        print(f"   {margin_name}: {n_sig} / {len(sig_mask)} components")
        if n_sig > 0:
            print(f"      Significant indices: {np.where(sig_mask)[0].tolist()}")


if __name__ == '__main__':
    # メインのデモ実行
    dpca, data, metadata = main()
    
    # オプション：有意性検定（時間がかかるためコメントアウト）
    # run_significance_analysis()
