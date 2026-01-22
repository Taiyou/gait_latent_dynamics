"""
Visualization Module for Demixed PCA Analysis

歩行ダイナミクス×メンタル状態のdPCA結果の可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class DPCAVisualizer:
    """
    dPCA結果の可視化クラス
    
    歩行ダイナミクスとメンタル状態の分析結果を
    多角的に可視化するためのツール群
    """
    
    # カラーパレット
    MENTAL_STATE_COLORS = {
        'neutral': '#7f7f7f',
        'anxious': '#d62728',
        'relaxed': '#2ca02c',
        'focused': '#1f77b4',
        'fatigued': '#ff7f0e',
        'energized': '#9467bd',
        'distracted': '#8c564b'
    }
    
    COMPONENT_COLORS = {
        'time': '#1f77b4',
        'condition': '#ff7f0e',
        'mental_state': '#ff7f0e',
        'interaction': '#2ca02c'
    }
    
    def __init__(
        self,
        style: str = 'seaborn-v0_8-whitegrid',
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100
    ):
        """
        Parameters
        ----------
        style : str
            Matplotlibのスタイル
        figsize : tuple
            デフォルトの図サイズ
        dpi : int
            解像度
        """
        self.figsize = figsize
        self.dpi = dpi
        
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn' in style else 'default')
    
    def plot_explained_variance(
        self,
        dpca_model,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        説明分散比を可視化
        
        Parameters
        ----------
        dpca_model : DemixedPCA
            フィット済みのdPCAモデル
        figsize : tuple, optional
            図サイズ
        save_path : str, optional
            保存先パス
            
        Returns
        -------
        fig : matplotlib.Figure
        """
        figsize = figsize or self.figsize
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=self.dpi)
        
        # 左: 各成分の累積説明分散比
        ax1 = axes[0]
        for margin_name, var_ratio in dpca_model.explained_variance_ratio_.items():
            cumsum = np.cumsum(var_ratio)
            color = self.COMPONENT_COLORS.get(margin_name, '#333333')
            ax1.plot(
                range(1, len(cumsum) + 1), 
                cumsum, 
                'o-', 
                label=margin_name,
                color=color,
                linewidth=2,
                markersize=8
            )
        
        ax1.set_xlabel('Number of Components', fontsize=12)
        ax1.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
        ax1.set_title('Cumulative Explained Variance by Marginalization', fontsize=14)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.set_xlim(0.5, len(cumsum) + 0.5)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        
        # 右: 総分散の分解（パイチャート）
        ax2 = axes[1]
        summary = dpca_model.get_demixing_summary()
        labels = list(summary.keys())
        sizes = list(summary.values())
        colors = [self.COMPONENT_COLORS.get(l, '#333333') for l in labels]
        
        wedges, texts, autotexts = ax2.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(labels),
            shadow=True,
            startangle=90
        )
        
        ax2.set_title('Variance Decomposition by Source', fontsize=14)
        
        for autotext in autotexts:
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_component_timecourse(
        self,
        dpca_model,
        X: np.ndarray,
        marginalization: str,
        components: List[int] = [0, 1, 2],
        mental_state_labels: Optional[List[str]] = None,
        time_axis: Optional[np.ndarray] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        各成分の時系列を条件別にプロット
        
        Parameters
        ----------
        dpca_model : DemixedPCA
            フィット済みのdPCAモデル
        X : ndarray
            入力データ
        marginalization : str
            プロットするmarginalization
        components : list of int
            プロットする成分のインデックス
        mental_state_labels : list of str, optional
            メンタル状態のラベル
        time_axis : ndarray, optional
            時間軸（歩行周期%）
        figsize : tuple, optional
            図サイズ
        save_path : str, optional
            保存先パス
        """
        figsize = figsize or (14, 4 * len(components))
        fig, axes = plt.subplots(
            len(components), 1, 
            figsize=figsize, 
            dpi=self.dpi,
            sharex=True
        )
        
        if len(components) == 1:
            axes = [axes]
        
        # 変換
        transformed = dpca_model.transform(X, marginalization=marginalization)
        n_comp, n_timepoints, n_conditions = transformed.shape
        
        # 時間軸
        if time_axis is None:
            time_axis = np.linspace(0, 100, n_timepoints)
        
        # ラベル
        if mental_state_labels is None:
            mental_state_labels = [f'Condition {i}' for i in range(n_conditions)]
        
        for ax_idx, comp_idx in enumerate(components):
            ax = axes[ax_idx]
            
            for cond_idx in range(n_conditions):
                label = mental_state_labels[cond_idx]
                color = self.MENTAL_STATE_COLORS.get(label, None)
                
                ax.plot(
                    time_axis,
                    transformed[comp_idx, :, cond_idx],
                    label=label,
                    color=color,
                    linewidth=2,
                    alpha=0.8
                )
            
            # 説明分散比を表示
            var_ratio = dpca_model.explained_variance_ratio_[marginalization][comp_idx]
            ax.set_title(
                f'{marginalization.capitalize()} Component {comp_idx + 1} '
                f'(Explained Variance: {var_ratio:.1%})',
                fontsize=12
            )
            ax.set_ylabel('Component Value', fontsize=10)
            ax.legend(loc='upper right', fontsize=9, ncol=2)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Gait Cycle (%)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_component_weights(
        self,
        dpca_model,
        marginalization: str,
        component: int = 0,
        feature_labels: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        成分の重みベクトルを可視化
        
        Parameters
        ----------
        dpca_model : DemixedPCA
            フィット済みのdPCAモデル
        marginalization : str
            対象のmarginalization
        component : int
            成分のインデックス
        feature_labels : list of str, optional
            特徴量ラベル
        figsize : tuple, optional
            図サイズ
        save_path : str, optional
            保存先パス
        """
        figsize = figsize or (10, 6)
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        weights = dpca_model.components_[marginalization][:, component]
        n_features = len(weights)
        
        if feature_labels is None:
            feature_labels = dpca_model._feature_names or \
                            [f'Feature {i}' for i in range(n_features)]
        
        # カラーマップ（正負で色分け）
        colors = ['#d62728' if w < 0 else '#2ca02c' for w in weights]
        
        y_pos = np.arange(n_features)
        ax.barh(y_pos, weights, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels, fontsize=10)
        ax.set_xlabel('Weight', fontsize=12)
        ax.set_title(
            f'{marginalization.capitalize()} Component {component + 1} - Feature Weights',
            fontsize=14
        )
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_mental_state_separation(
        self,
        dpca_model,
        X: np.ndarray,
        mental_state_labels: List[str],
        components: Tuple[int, int] = (0, 1),
        marginalization: str = 'condition',
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        メンタル状態の分離を2D散布図で可視化
        
        Parameters
        ----------
        dpca_model : DemixedPCA
            フィット済みのdPCAモデル
        X : ndarray
            入力データ
        mental_state_labels : list of str
            メンタル状態ラベル
        components : tuple of int
            プロットする2つの成分
        marginalization : str
            使用するmarginalization
        figsize : tuple, optional
            図サイズ
        save_path : str, optional
            保存先パス
        """
        figsize = figsize or (10, 8)
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # 変換
        transformed = dpca_model.transform(X, marginalization=marginalization)
        
        # 各条件の平均を計算
        n_conditions = transformed.shape[2]
        
        for cond_idx in range(n_conditions):
            label = mental_state_labels[cond_idx]
            color = self.MENTAL_STATE_COLORS.get(label, None)
            
            # 時間方向の軌跡をプロット
            x = transformed[components[0], :, cond_idx]
            y = transformed[components[1], :, cond_idx]
            
            ax.plot(x, y, '-', color=color, alpha=0.7, linewidth=2)
            ax.scatter(
                x, y, 
                c=[color] * len(x), 
                s=30, 
                alpha=0.5
            )
            
            # 平均点にラベル
            mean_x, mean_y = x.mean(), y.mean()
            ax.scatter(mean_x, mean_y, c=color, s=200, marker='*', 
                      edgecolors='black', linewidths=1, zorder=10)
            ax.annotate(
                label, 
                (mean_x, mean_y),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=11,
                fontweight='bold'
            )
        
        ax.set_xlabel(
            f'{marginalization.capitalize()} Component {components[0] + 1}',
            fontsize=12
        )
        ax.set_ylabel(
            f'{marginalization.capitalize()} Component {components[1] + 1}',
            fontsize=12
        )
        ax.set_title(
            f'Mental State Separation in {marginalization.capitalize()} Space',
            fontsize=14
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_comprehensive_summary(
        self,
        dpca_model,
        X: np.ndarray,
        mental_state_labels: List[str],
        feature_labels: Optional[List[str]] = None,
        time_axis: Optional[np.ndarray] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        包括的なサマリーを1枚の図にプロット
        
        Parameters
        ----------
        dpca_model : DemixedPCA
            フィット済みのdPCAモデル
        X : ndarray
            入力データ
        mental_state_labels : list of str
            メンタル状態ラベル
        feature_labels : list of str, optional
            特徴量ラベル
        time_axis : ndarray, optional
            時間軸
        figsize : tuple, optional
            図サイズ
        save_path : str, optional
            保存先パス
        """
        figsize = figsize or (18, 14)
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # 1. 説明分散比（左上）
        ax1 = fig.add_subplot(gs[0, 0])
        for margin_name, var_ratio in dpca_model.explained_variance_ratio_.items():
            cumsum = np.cumsum(var_ratio)
            color = self.COMPONENT_COLORS.get(margin_name, '#333333')
            ax1.plot(range(1, len(cumsum) + 1), cumsum, 'o-', 
                    label=margin_name, color=color, linewidth=2)
        ax1.set_xlabel('# Components')
        ax1.set_ylabel('Cumulative Var. Ratio')
        ax1.set_title('Explained Variance', fontsize=12)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. 分散分解パイチャート（中央上）
        ax2 = fig.add_subplot(gs[0, 1])
        summary = dpca_model.get_demixing_summary()
        labels = list(summary.keys())
        sizes = list(summary.values())
        colors = [self.COMPONENT_COLORS.get(l, '#333333') for l in labels]
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Variance Decomposition', fontsize=12)
        
        # 3. メンタル状態分離（右上）
        ax3 = fig.add_subplot(gs[0, 2])
        margin_key = 'condition' if 'condition' in dpca_model.components_ else 'mental_state'
        transformed = dpca_model.transform(X, marginalization=margin_key)
        
        for cond_idx in range(len(mental_state_labels)):
            label = mental_state_labels[cond_idx]
            color = self.MENTAL_STATE_COLORS.get(label, None)
            x = transformed[0, :, cond_idx]
            y = transformed[1, :, cond_idx]
            ax3.plot(x, y, '-', color=color, alpha=0.7, linewidth=1.5, label=label)
        ax3.set_xlabel('Component 1')
        ax3.set_ylabel('Component 2')
        ax3.set_title('Mental State Separation', fontsize=12)
        ax3.legend(fontsize=8, loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4-6. 各marginalizationの第1成分時系列
        n_timepoints = X.shape[1] if X.ndim == 3 else X.shape[2]
        if time_axis is None:
            time_axis = np.linspace(0, 100, n_timepoints)
        
        marginalizations = list(dpca_model.components_.keys())
        
        for i, margin_name in enumerate(marginalizations[:3]):
            ax = fig.add_subplot(gs[1, i])
            transformed = dpca_model.transform(X, marginalization=margin_name)
            
            for cond_idx in range(len(mental_state_labels)):
                label = mental_state_labels[cond_idx]
                color = self.MENTAL_STATE_COLORS.get(label, None)
                ax.plot(time_axis, transformed[0, :, cond_idx], 
                       color=color, linewidth=1.5, label=label)
            
            var_ratio = dpca_model.explained_variance_ratio_[margin_name][0]
            ax.set_title(f'{margin_name.capitalize()} C1 ({var_ratio:.1%})', fontsize=11)
            ax.set_xlabel('Gait Cycle (%)')
            ax.set_ylabel('Value')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # 7-9. 成分重み
        if feature_labels is None:
            feature_labels = dpca_model._feature_names or \
                            [f'F{i}' for i in range(X.shape[0] if X.ndim == 3 else X.shape[1])]
        
        for i, margin_name in enumerate(marginalizations[:3]):
            ax = fig.add_subplot(gs[2, i])
            weights = dpca_model.components_[margin_name][:, 0]
            colors = ['#d62728' if w < 0 else '#2ca02c' for w in weights]
            
            y_pos = np.arange(len(weights))
            ax.barh(y_pos, weights, color=colors, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_labels, fontsize=8)
            ax.set_title(f'{margin_name.capitalize()} C1 Weights', fontsize=11)
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
            ax.grid(True, axis='x', alpha=0.3)
        
        plt.suptitle(
            'Demixed PCA Analysis: Gait Dynamics × Mental State',
            fontsize=16, fontweight='bold', y=1.02
        )
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_reconstruction_quality(
        self,
        dpca_model,
        X: np.ndarray,
        feature_idx: int = 0,
        condition_idx: int = 0,
        time_axis: Optional[np.ndarray] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        再構成品質を可視化
        
        Parameters
        ----------
        dpca_model : DemixedPCA
            フィット済みのdPCAモデル
        X : ndarray
            元データ
        feature_idx : int
            表示する特徴量のインデックス
        condition_idx : int
            表示する条件のインデックス
        time_axis : ndarray, optional
            時間軸
        figsize : tuple, optional
            図サイズ
        save_path : str, optional
            保存先パス
        """
        figsize = figsize or (12, 6)
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=self.dpi)
        
        X_3d = dpca_model._validate_and_reshape_data(X)
        n_timepoints = X_3d.shape[1]
        
        if time_axis is None:
            time_axis = np.linspace(0, 100, n_timepoints)
        
        # 変換と逆変換
        transformed = dpca_model.transform(X)
        reconstructed = dpca_model.inverse_transform(transformed)
        
        # 左: 元データ vs 再構成データ
        ax1 = axes[0]
        original = X_3d[feature_idx, :, condition_idx]
        recon = reconstructed[feature_idx, :, condition_idx]
        
        ax1.plot(time_axis, original, 'b-', label='Original', linewidth=2)
        ax1.plot(time_axis, recon, 'r--', label='Reconstructed', linewidth=2)
        ax1.fill_between(
            time_axis, original, recon, 
            alpha=0.3, color='gray', label='Difference'
        )
        
        ax1.set_xlabel('Gait Cycle (%)')
        ax1.set_ylabel('Value')
        ax1.set_title('Original vs Reconstructed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右: 各成分の寄与
        ax2 = axes[1]
        
        for margin_name in dpca_model.components_.keys():
            trans_single = {margin_name: transformed[margin_name]}
            recon_single = dpca_model.inverse_transform(trans_single, marginalization=margin_name)
            contribution = recon_single[feature_idx, :, condition_idx] - dpca_model.mean_[feature_idx]
            
            color = self.COMPONENT_COLORS.get(margin_name, '#333333')
            ax2.plot(time_axis, contribution, label=margin_name, color=color, linewidth=2)
        
        ax2.set_xlabel('Gait Cycle (%)')
        ax2.set_ylabel('Contribution')
        ax2.set_title('Component Contributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def create_interactive_plot(
        self,
        dpca_model,
        X: np.ndarray,
        mental_state_labels: List[str],
        feature_labels: Optional[List[str]] = None
    ):
        """
        Plotlyを使用したインタラクティブプロットを作成
        
        Parameters
        ----------
        dpca_model : DemixedPCA
            フィット済みのdPCAモデル
        X : ndarray
            入力データ
        mental_state_labels : list of str
            メンタル状態ラベル
        feature_labels : list of str, optional
            特徴量ラベル
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("Plotly is required for interactive plots. Install with: pip install plotly")
        
        # サブプロットを作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Explained Variance',
                'Mental State Separation',
                'Time Component',
                'Condition Component'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 1. 説明分散
        for margin_name, var_ratio in dpca_model.explained_variance_ratio_.items():
            cumsum = np.cumsum(var_ratio)
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cumsum) + 1)),
                    y=cumsum,
                    mode='lines+markers',
                    name=margin_name
                ),
                row=1, col=1
            )
        
        # 2. メンタル状態分離
        margin_key = 'condition' if 'condition' in dpca_model.components_ else 'mental_state'
        transformed = dpca_model.transform(X, marginalization=margin_key)
        
        for cond_idx, label in enumerate(mental_state_labels):
            x = transformed[0, :, cond_idx]
            y = transformed[1, :, cond_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='lines+markers',
                    name=label,
                    legendgroup=label
                ),
                row=1, col=2
            )
        
        # 3. 時間成分
        time_trans = dpca_model.transform(X, marginalization='time')
        time_axis = np.linspace(0, 100, time_trans.shape[1])
        
        for cond_idx, label in enumerate(mental_state_labels):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=time_trans[0, :, cond_idx],
                    mode='lines',
                    name=label,
                    legendgroup=label,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. 条件成分
        cond_trans = dpca_model.transform(X, marginalization=margin_key)
        
        for cond_idx, label in enumerate(mental_state_labels):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=cond_trans[0, :, cond_idx],
                    mode='lines',
                    name=label,
                    legendgroup=label,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Demixed PCA Analysis: Gait × Mental State",
            height=800,
            showlegend=True
        )
        
        return fig


def quick_plot_dpca_results(
    dpca_model,
    X: np.ndarray,
    mental_state_labels: List[str],
    feature_labels: Optional[List[str]] = None,
    save_dir: Optional[str] = None
) -> List[plt.Figure]:
    """
    dPCA結果のクイックプロット
    
    Parameters
    ----------
    dpca_model : DemixedPCA
        フィット済みのdPCAモデル
    X : ndarray
        入力データ
    mental_state_labels : list of str
        メンタル状態ラベル
    feature_labels : list of str, optional
        特徴量ラベル
    save_dir : str, optional
        保存ディレクトリ
        
    Returns
    -------
    figures : list of matplotlib.Figure
    """
    viz = DPCAVisualizer()
    figures = []
    
    # 保存ディレクトリの作成
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 包括的サマリー
    fig1 = viz.plot_comprehensive_summary(
        dpca_model, X, mental_state_labels, feature_labels,
        save_path=str(save_dir / 'summary.png') if save_dir else None
    )
    figures.append(fig1)
    
    # 2. 説明分散
    fig2 = viz.plot_explained_variance(
        dpca_model,
        save_path=str(save_dir / 'explained_variance.png') if save_dir else None
    )
    figures.append(fig2)
    
    # 3. メンタル状態分離
    margin_key = 'condition' if 'condition' in dpca_model.components_ else 'mental_state'
    fig3 = viz.plot_mental_state_separation(
        dpca_model, X, mental_state_labels,
        marginalization=margin_key,
        save_path=str(save_dir / 'mental_state_separation.png') if save_dir else None
    )
    figures.append(fig3)
    
    return figures
