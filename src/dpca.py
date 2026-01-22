"""
Demixed Principal Component Analysis (dPCA) Implementation

歩行ダイナミクスとメンタル状態の関係を分析するためのdPCA実装

References:
- Kobak et al. (2016) "Demixed principal component analysis of neural population data"
- https://github.com/machenslab/dPCA
"""

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Optional, Tuple, Union
from itertools import combinations
import warnings


class DemixedPCA(BaseEstimator, TransformerMixin):
    """
    Demixed Principal Component Analysis for Gait-Mental State Analysis
    
    歩行キネマティクス/キネティクスデータをメンタル状態と時間の
    成分に分離するdPCA実装
    
    Parameters
    ----------
    n_components : int, default=10
        各marginalizationで抽出する主成分の数
    regularizer : float or 'auto', default='auto'
        正則化パラメータ。'auto'の場合は交差検証で決定
    n_splits : int, default=5
        交差検証の分割数（正則化パラメータ決定用）
    
    Attributes
    ----------
    components_ : dict
        各marginalizationの主成分ベクトル
    explained_variance_ : dict
        各成分の説明分散
    explained_variance_ratio_ : dict
        各成分の説明分散比
    mean_ : ndarray
        全体の平均
    marginalized_means_ : dict
        各条件での平均化されたデータ
    """
    
    def __init__(
        self,
        n_components: int = 10,
        regularizer: Union[float, str] = 'auto',
        n_splits: int = 5
    ):
        self.n_components = n_components
        self.regularizer = regularizer
        self.n_splits = n_splits
        
        # Fitted attributes
        self.components_: Dict[str, np.ndarray] = {}
        self.explained_variance_: Dict[str, np.ndarray] = {}
        self.explained_variance_ratio_: Dict[str, np.ndarray] = {}
        self.mean_: Optional[np.ndarray] = None
        self.marginalized_means_: Dict[str, np.ndarray] = {}
        self._encoder: Optional[Dict[str, np.ndarray]] = None
        self._decoder: Optional[Dict[str, np.ndarray]] = None
        self._regularizer_value: float = 0.0
        self._feature_names: List[str] = []
        self._marginalization_labels: List[str] = []
        
    def fit(
        self,
        X: np.ndarray,
        feature_labels: Optional[List[str]] = None,
        marginalization_labels: Optional[List[str]] = None
    ) -> 'DemixedPCA':
        """
        dPCAモデルをフィット
        
        Parameters
        ----------
        X : ndarray of shape (n_trials, n_features, n_timepoints, n_conditions)
            or (n_features, n_timepoints, n_conditions) for trial-averaged data
            入力データ。歩行特徴量 × 時間 × メンタル状態条件
        feature_labels : list of str, optional
            特徴量のラベル（例: ['hip_angle', 'knee_angle', ...]）
        marginalization_labels : list of str, optional
            条件のラベル（例: ['time', 'mental_state', 'interaction']）
            
        Returns
        -------
        self : DemixedPCA
            フィットされたモデル
        """
        # データ形状の確認と変換
        X = self._validate_and_reshape_data(X)
        
        n_features, n_timepoints, n_conditions = X.shape
        
        # 特徴量ラベルの設定
        if feature_labels is not None:
            self._feature_names = feature_labels
        else:
            self._feature_names = [f'feature_{i}' for i in range(n_features)]
            
        # Marginalizationラベルの設定
        if marginalization_labels is not None:
            self._marginalization_labels = marginalization_labels
        else:
            self._marginalization_labels = ['time', 'condition', 'interaction']
        
        # 全体平均を計算
        self.mean_ = X.mean(axis=(1, 2))
        
        # データを中心化
        X_centered = X - self.mean_[:, np.newaxis, np.newaxis]
        
        # Marginalized averagesを計算
        self._compute_marginalized_means(X_centered)
        
        # 正則化パラメータの決定
        if self.regularizer == 'auto':
            self._regularizer_value = self._find_optimal_regularizer(X_centered)
        else:
            self._regularizer_value = float(self.regularizer)
        
        # 各marginalizationに対してdPCAを実行
        self._fit_dpca(X_centered)
        
        return self
    
    def _validate_and_reshape_data(self, X: np.ndarray) -> np.ndarray:
        """データの形状を検証し、必要に応じてリシェイプ"""
        if X.ndim == 4:
            # (n_trials, n_features, n_timepoints, n_conditions)
            # → trial average to (n_features, n_timepoints, n_conditions)
            return X.mean(axis=0)
        elif X.ndim == 3:
            return X
        else:
            raise ValueError(
                f"Expected 3D or 4D array, got {X.ndim}D array. "
                "Shape should be (n_features, n_timepoints, n_conditions) or "
                "(n_trials, n_features, n_timepoints, n_conditions)"
            )
    
    def _compute_marginalized_means(self, X: np.ndarray) -> None:
        """
        各marginalization（周辺化）に対する平均を計算
        
        Marginalizationの種類:
        - time: 時間方向に平均化（条件依存成分）
        - condition: 条件方向に平均化（時間依存成分）  
        - interaction: 上記を引いた残差（相互作用成分）
        """
        n_features, n_timepoints, n_conditions = X.shape
        
        # 時間方向の平均（各条件での平均 → 条件依存成分）
        mean_over_time = X.mean(axis=1, keepdims=True)
        
        # 条件方向の平均（各時点での平均 → 時間依存成分）
        mean_over_condition = X.mean(axis=2, keepdims=True)
        
        # 全体平均
        grand_mean = X.mean(axis=(1, 2), keepdims=True)
        
        # Marginalized components
        # time: 時間依存成分（条件を平均化）
        self.marginalized_means_['time'] = mean_over_condition - grand_mean
        
        # condition: 条件依存成分（時間を平均化）= メンタル状態依存
        self.marginalized_means_['condition'] = mean_over_time - grand_mean
        
        # interaction: 相互作用成分
        self.marginalized_means_['interaction'] = (
            X - mean_over_time - mean_over_condition + grand_mean
        )
    
    def _find_optimal_regularizer(self, X: np.ndarray) -> float:
        """交差検証で最適な正則化パラメータを決定"""
        n_features = X.shape[0]
        
        # 共分散行列を計算
        X_flat = X.reshape(n_features, -1)
        C = np.cov(X_flat)
        
        # 正則化の候補値
        reg_values = np.logspace(-6, 0, 20) * np.trace(C) / n_features
        
        # 簡易的な正則化選択（データの特異値に基づく）
        try:
            _, s, _ = np.linalg.svd(C, full_matrices=False)
            # 条件数が大きすぎる場合は正則化
            condition_number = s[0] / s[-1] if s[-1] > 0 else np.inf
            
            if condition_number > 1e10:
                # 適度な正則化
                optimal_reg = np.median(s) * 0.01
            else:
                optimal_reg = 1e-10
        except np.linalg.LinAlgError:
            optimal_reg = 1e-6 * np.trace(C) / n_features
            
        return optimal_reg
    
    def _fit_dpca(self, X: np.ndarray) -> None:
        """dPCAのメイン計算"""
        n_features, n_timepoints, n_conditions = X.shape
        
        # フラット化したデータの共分散行列
        X_flat = X.reshape(n_features, -1)
        C_total = np.cov(X_flat)
        
        # 正則化
        C_total_reg = C_total + self._regularizer_value * np.eye(n_features)
        
        # 各marginalizationに対してdPCAを実行
        for margin_name, X_margin in self.marginalized_means_.items():
            # Marginalized dataをフラット化
            X_margin_flat = X_margin.reshape(n_features, -1)
            
            # Marginalized共分散行列
            C_margin = np.cov(X_margin_flat)
            
            # 一般化固有値問題を解く: C_margin * v = lambda * C_total * v
            # これにより、marginalized varianceを最大化する方向を見つける
            try:
                eigenvalues, eigenvectors = linalg.eigh(
                    C_margin, 
                    C_total_reg
                )
            except np.linalg.LinAlgError:
                warnings.warn(
                    f"Eigenvalue decomposition failed for {margin_name}. "
                    "Using standard PCA instead."
                )
                eigenvalues, eigenvectors = linalg.eigh(C_margin)
            
            # 固有値の大きい順にソート
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # n_components分だけ保持
            n_comp = min(self.n_components, n_features)
            
            # Decoder（dPCA成分）
            decoder = eigenvectors[:, :n_comp]
            
            # Encoder（pseudoinverse for reconstruction）
            # D^T * C_total で近似
            encoder = linalg.solve(C_total_reg, decoder).T
            
            # 正規化
            for i in range(n_comp):
                norm = np.dot(encoder[i], np.dot(C_total_reg, decoder[:, i]))
                if norm > 0:
                    encoder[i] /= np.sqrt(norm)
                    decoder[:, i] /= np.sqrt(norm)
            
            # 結果を保存
            self.components_[margin_name] = decoder
            self.explained_variance_[margin_name] = eigenvalues[:n_comp]
            
            # 説明分散比を計算
            total_var = np.trace(C_margin) if np.trace(C_margin) > 0 else 1.0
            self.explained_variance_ratio_[margin_name] = eigenvalues[:n_comp] / total_var
            
        # EncoderとDecoderを保存
        self._decoder = self.components_.copy()
        self._encoder = {}
        for margin_name in self.components_:
            self._encoder[margin_name] = linalg.pinv(self.components_[margin_name]).T
    
    def transform(
        self,
        X: np.ndarray,
        marginalization: Optional[str] = None
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        データを低次元空間に射影
        
        Parameters
        ----------
        X : ndarray of shape (n_features, n_timepoints, n_conditions)
            射影するデータ
        marginalization : str, optional
            特定のmarginalizationのみを返す場合に指定
            
        Returns
        -------
        transformed : dict or ndarray
            射影されたデータ。marginalizationを指定した場合はndarray、
            指定しない場合はdict
        """
        X = self._validate_and_reshape_data(X)
        
        # 中心化
        X_centered = X - self.mean_[:, np.newaxis, np.newaxis]
        n_features, n_timepoints, n_conditions = X_centered.shape
        
        if marginalization is not None:
            if marginalization not in self.components_:
                raise ValueError(
                    f"Unknown marginalization: {marginalization}. "
                    f"Available: {list(self.components_.keys())}"
                )
            # 特定のmarginalizationのみ
            decoder = self.components_[marginalization]
            X_flat = X_centered.reshape(n_features, -1)
            transformed = decoder.T @ X_flat
            return transformed.reshape(-1, n_timepoints, n_conditions)
        
        # 全てのmarginalization
        transformed = {}
        X_flat = X_centered.reshape(n_features, -1)
        
        for margin_name, decoder in self.components_.items():
            proj = decoder.T @ X_flat
            transformed[margin_name] = proj.reshape(-1, n_timepoints, n_conditions)
            
        return transformed
    
    def inverse_transform(
        self,
        X_transformed: Union[Dict[str, np.ndarray], np.ndarray],
        marginalization: Optional[str] = None
    ) -> np.ndarray:
        """
        低次元表現から元の空間に再構成
        
        Parameters
        ----------
        X_transformed : dict or ndarray
            射影されたデータ
        marginalization : str, optional
            特定のmarginalizationを再構成する場合に指定
            
        Returns
        -------
        X_reconstructed : ndarray
            再構成されたデータ
        """
        if marginalization is not None:
            if isinstance(X_transformed, dict):
                X_trans = X_transformed[marginalization]
            else:
                X_trans = X_transformed
                
            decoder = self.components_[marginalization]
            n_comp, n_timepoints, n_conditions = X_trans.shape
            X_flat = X_trans.reshape(n_comp, -1)
            reconstructed = decoder @ X_flat
            return reconstructed.reshape(-1, n_timepoints, n_conditions) + \
                   self.mean_[:, np.newaxis, np.newaxis]
        
        # 全てのmarginalizationの合計
        if not isinstance(X_transformed, dict):
            raise ValueError(
                "X_transformed must be a dict when marginalization is not specified"
            )
            
        # 最初の要素から形状を取得
        first_key = list(X_transformed.keys())[0]
        _, n_timepoints, n_conditions = X_transformed[first_key].shape
        n_features = len(self.mean_)
        
        reconstructed = np.zeros((n_features, n_timepoints, n_conditions))
        
        for margin_name, X_trans in X_transformed.items():
            decoder = self.components_[margin_name]
            X_flat = X_trans.reshape(X_trans.shape[0], -1)
            reconstructed += (decoder @ X_flat).reshape(n_features, n_timepoints, n_conditions)
        
        return reconstructed + self.mean_[:, np.newaxis, np.newaxis]
    
    def get_demixing_summary(self) -> Dict[str, float]:
        """
        各marginalizationの寄与度のサマリーを取得
        
        Returns
        -------
        summary : dict
            各marginalizationの累積説明分散比
        """
        summary = {}
        for margin_name, var_ratio in self.explained_variance_ratio_.items():
            summary[margin_name] = float(var_ratio.sum())
        return summary
    
    def get_component_timecourse(
        self,
        X: np.ndarray,
        marginalization: str,
        component: int = 0
    ) -> np.ndarray:
        """
        特定の成分の時系列を取得
        
        Parameters
        ----------
        X : ndarray
            入力データ
        marginalization : str
            対象のmarginalization
        component : int
            成分のインデックス
            
        Returns
        -------
        timecourse : ndarray of shape (n_timepoints, n_conditions)
            各条件での時系列
        """
        transformed = self.transform(X, marginalization=marginalization)
        return transformed[component]
    
    def significance_analysis(
        self,
        X: np.ndarray,
        n_shuffles: int = 1000,
        alpha: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        シャッフルテストによる有意性検定
        
        Parameters
        ----------
        X : ndarray
            入力データ (n_trials, n_features, n_timepoints, n_conditions)
        n_shuffles : int
            シャッフル回数
        alpha : float
            有意水準
            
        Returns
        -------
        significance : dict
            各成分の有意性マスク
        """
        if X.ndim != 4:
            raise ValueError(
                "Significance analysis requires trial-level data "
                "(n_trials, n_features, n_timepoints, n_conditions)"
            )
        
        n_trials, n_features, n_timepoints, n_conditions = X.shape
        
        # 元のデータでの変換
        X_mean = X.mean(axis=0)
        transformed_orig = self.transform(X_mean)
        
        significance = {}
        
        for margin_name in self.components_:
            orig_var = np.var(transformed_orig[margin_name], axis=(1, 2))
            n_comp = len(orig_var)
            shuffle_vars = np.zeros((n_shuffles, n_comp))
            
            for i in range(n_shuffles):
                # 条件ラベルをシャッフル
                X_shuffled = X.copy()
                for trial in range(n_trials):
                    perm = np.random.permutation(n_conditions)
                    X_shuffled[trial] = X_shuffled[trial][:, :, perm]
                
                X_shuffled_mean = X_shuffled.mean(axis=0)
                transformed_shuffle = self.transform(X_shuffled_mean)
                shuffle_vars[i] = np.var(
                    transformed_shuffle[margin_name], axis=(1, 2)
                )
            
            # p値の計算
            p_values = np.mean(shuffle_vars >= orig_var, axis=0)
            significance[margin_name] = p_values < alpha
            
        return significance


class GaitDPCA(DemixedPCA):
    """
    歩行解析に特化したdPCA拡張クラス
    
    歩行周期の位相、メンタル状態、およびそれらの相互作用を
    明示的に分離するための追加機能を提供
    """
    
    GAIT_FEATURES = [
        'hip_flexion', 'hip_abduction', 'hip_rotation',
        'knee_flexion', 'knee_rotation',
        'ankle_dorsiflexion', 'ankle_inversion',
        'pelvis_tilt', 'pelvis_obliquity', 'pelvis_rotation',
        'trunk_flexion', 'trunk_lateral', 'trunk_rotation',
        'stride_length', 'step_width', 'cadence',
        'grf_vertical', 'grf_anterior', 'grf_lateral'
    ]
    
    MENTAL_STATES = [
        'neutral', 'anxious', 'relaxed', 'focused', 
        'distracted', 'fatigued', 'energized'
    ]
    
    def __init__(
        self,
        n_components: int = 10,
        regularizer: Union[float, str] = 'auto',
        n_splits: int = 5,
        gait_cycle_normalize: bool = True
    ):
        super().__init__(
            n_components=n_components,
            regularizer=regularizer,
            n_splits=n_splits
        )
        self.gait_cycle_normalize = gait_cycle_normalize
        self._mental_state_mapping: Dict[str, int] = {}
        
    def fit_with_labels(
        self,
        X: np.ndarray,
        mental_state_labels: List[str],
        feature_labels: Optional[List[str]] = None
    ) -> 'GaitDPCA':
        """
        メンタル状態ラベル付きでフィット
        
        Parameters
        ----------
        X : ndarray of shape (n_features, n_timepoints, n_conditions)
            入力データ
        mental_state_labels : list of str
            各条件に対応するメンタル状態ラベル
        feature_labels : list of str, optional
            特徴量ラベル
        """
        # メンタル状態マッピングを作成
        unique_states = list(set(mental_state_labels))
        self._mental_state_mapping = {
            state: i for i, state in enumerate(unique_states)
        }
        
        # 標準のfitを呼び出し
        marginalization_labels = ['time', 'mental_state', 'interaction']
        return self.fit(
            X,
            feature_labels=feature_labels,
            marginalization_labels=marginalization_labels
        )
    
    def get_mental_state_components(self) -> np.ndarray:
        """メンタル状態に関連する成分を取得"""
        return self.components_.get('condition', self.components_.get('mental_state'))
    
    def get_gait_phase_components(self) -> np.ndarray:
        """歩行位相（時間）に関連する成分を取得"""
        return self.components_.get('time')
    
    def get_interaction_components(self) -> np.ndarray:
        """相互作用成分を取得"""
        return self.components_.get('interaction')
    
    def analyze_mental_state_separation(
        self,
        X: np.ndarray,
        mental_state_labels: List[str]
    ) -> Dict[str, float]:
        """
        メンタル状態間の分離度を分析
        
        Returns
        -------
        separation_scores : dict
            各メンタル状態ペアの分離度スコア
        """
        transformed = self.transform(X, marginalization='condition')
        
        unique_states = list(set(mental_state_labels))
        separation_scores = {}
        
        for state1, state2 in combinations(unique_states, 2):
            idx1 = mental_state_labels.index(state1)
            idx2 = mental_state_labels.index(state2)
            
            # 各状態の平均を計算
            mean1 = transformed[:, :, idx1].mean(axis=1)
            mean2 = transformed[:, :, idx2].mean(axis=1)
            
            # ユークリッド距離を計算
            distance = np.linalg.norm(mean1 - mean2)
            
            key = f"{state1}_vs_{state2}"
            separation_scores[key] = float(distance)
        
        return separation_scores


class ContinuousScoreDPCA:
    """
    連続値メンタルスコア（wellbeing等）対応のdPCA拡張
    
    メンタル状態が時間に依存せず一定のスコア（例：wellbeingスコア）で
    表される場合に使用します。
    
    Parameters
    ----------
    n_components : int, default=10
        抽出する成分数
    
    Attributes
    ----------
    gait_components_ : ndarray
        歩行パターンの主成分（時間依存）
    score_weights_ : ndarray
        メンタルスコアとの関連を表す重み
    correlation_ : ndarray
        各成分とメンタルスコアの相関
    
    Notes
    -----
    このクラスは以下のモデルを仮定します：
    
    歩行データ X(t) = 時間成分(t) + スコア依存成分 * wellbeing_score + 残差
    
    wellbeingスコアは各トライアル/条件で一定値を取り、
    歩行周期内では変化しません。
    """
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.gait_components_: Optional[np.ndarray] = None
        self.score_weights_: Optional[np.ndarray] = None
        self.correlation_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self._feature_names: List[str] = []
    
    def fit(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        feature_labels: Optional[List[str]] = None
    ) -> 'ContinuousScoreDPCA':
        """
        連続値スコアを用いてフィット
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, n_timepoints)
            歩行データ。n_samplesは各トライアル/被験者
        scores : ndarray of shape (n_samples,)
            各サンプルのメンタルスコア（wellbeing等）
            時間に依存せず、各サンプルで一定値
        feature_labels : list of str, optional
            特徴量ラベル
            
        Returns
        -------
        self : ContinuousScoreDPCA
        
        Examples
        --------
        >>> # 30人の被験者、各15特徴量、100時間点
        >>> X = np.random.randn(30, 15, 100)
        >>> wellbeing_scores = np.random.rand(30) * 10  # 0-10のスコア
        >>> model = ContinuousScoreDPCA(n_components=5)
        >>> model.fit(X, wellbeing_scores)
        """
        n_samples, n_features, n_timepoints = X.shape
        
        if feature_labels is not None:
            self._feature_names = feature_labels
        else:
            self._feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # スコアを正規化
        scores_normalized = (scores - scores.mean()) / (scores.std() + 1e-10)
        
        # 全体平均
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        # 時間方向に平均化して、スコア依存成分を抽出
        # X_time_avg: (n_samples, n_features)
        X_time_avg = X_centered.mean(axis=2)
        
        # スコアとの相関を計算
        # 各特徴量とスコアの相関
        correlations = np.array([
            np.corrcoef(X_time_avg[:, f], scores_normalized)[0, 1]
            for f in range(n_features)
        ])
        
        # スコア依存成分を回帰で推定
        # X_time_avg = beta * scores + residual
        scores_design = scores_normalized[:, np.newaxis]
        beta = np.linalg.lstsq(scores_design, X_time_avg, rcond=None)[0]
        self.score_weights_ = beta.flatten()
        
        # 残差から時間依存成分をPCAで抽出
        X_residual = X_centered - np.outer(scores_normalized, self.score_weights_)[:, :, np.newaxis]
        
        # 時間方向の共分散を計算
        X_flat = X_residual.reshape(n_samples, -1)
        cov = np.cov(X_flat.T)
        
        # PCAで時間依存成分を抽出
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        
        n_comp = min(self.n_components, len(eigenvalues))
        self.gait_components_ = eigenvectors[:, idx[:n_comp]]
        self.explained_variance_ = eigenvalues[idx[:n_comp]]
        
        # 各成分とスコアの相関を計算
        projections = X_flat @ self.gait_components_
        self.correlation_ = np.array([
            np.corrcoef(projections[:, i], scores_normalized)[0, 1]
            for i in range(n_comp)
        ])
        
        return self
    
    def transform(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        データを変換
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, n_timepoints)
            
        Returns
        -------
        transformed : dict
            'time': 時間依存成分への射影
            'score_effect': スコア依存成分の推定値
        """
        n_samples = X.shape[0]
        X_centered = X - self.mean_
        X_flat = X_centered.reshape(n_samples, -1)
        
        return {
            'time': X_flat @ self.gait_components_,
            'score_effect': X_centered.mean(axis=2) @ self.score_weights_
        }
    
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        歩行データからメンタルスコアを予測
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, n_timepoints)
            
        Returns
        -------
        predicted_scores : ndarray of shape (n_samples,)
            予測されたメンタルスコア
        """
        X_centered = X - self.mean_
        X_time_avg = X_centered.mean(axis=2)
        
        # 重みを使って予測
        weight_norm = np.linalg.norm(self.score_weights_)
        if weight_norm > 0:
            predicted = X_time_avg @ self.score_weights_ / (weight_norm ** 2)
        else:
            predicted = np.zeros(X.shape[0])
        
        return predicted
    
    def get_score_related_features(self, threshold: float = 0.3) -> List[str]:
        """
        メンタルスコアと強く関連する特徴量を取得
        
        Parameters
        ----------
        threshold : float
            相関の閾値（絶対値）
            
        Returns
        -------
        features : list of str
            関連する特徴量のリスト
        """
        # 各特徴量の重みの絶対値でソート
        weights_abs = np.abs(self.score_weights_)
        weights_normalized = weights_abs / (weights_abs.max() + 1e-10)
        
        related = [
            self._feature_names[i]
            for i in range(len(self._feature_names))
            if weights_normalized[i] > threshold
        ]
        
        return related
    
    def summary(self) -> Dict:
        """
        分析結果のサマリーを取得
        
        Returns
        -------
        summary : dict
            分析結果のサマリー
        """
        weights_abs = np.abs(self.score_weights_)
        top_indices = np.argsort(weights_abs)[::-1][:5]
        
        return {
            'top_score_related_features': [
                (self._feature_names[i], float(self.score_weights_[i]))
                for i in top_indices
            ],
            'explained_variance_time': float(self.explained_variance_.sum()),
            'n_components': self.n_components
        }
