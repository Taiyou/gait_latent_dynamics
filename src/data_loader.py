"""
Gait Data Loader and Synthetic Data Generator

歩行データの読み込みとシミュレーションデータ生成
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json


def generate_synthetic_gait_data(
    n_trials: int = 50,
    n_features: int = 15,
    n_timepoints: int = 100,
    n_conditions: int = 5,
    mental_state_effect_strength: float = 0.3,
    time_effect_strength: float = 0.5,
    interaction_strength: float = 0.15,
    noise_level: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    デモ用の合成歩行データを生成
    
    メンタル状態と歩行ダイナミクスの関係をシミュレート
    
    Parameters
    ----------
    n_trials : int
        トライアル数
    n_features : int
        歩行特徴量の数（関節角度、GRFなど）
    n_timepoints : int
        1歩行周期の時間点数（0-100% gait cycle）
    n_conditions : int
        メンタル状態の条件数
    mental_state_effect_strength : float
        メンタル状態の影響の強さ
    time_effect_strength : float
        時間（歩行位相）の影響の強さ
    interaction_strength : float
        相互作用の強さ
    noise_level : float
        ノイズレベル
    random_state : int, optional
        乱数シード
        
    Returns
    -------
    data : ndarray of shape (n_trials, n_features, n_timepoints, n_conditions)
        生成されたデータ
    metadata : dict
        メタデータ（ラベル情報など）
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 特徴量ラベル
    feature_labels = [
        'hip_flexion', 'hip_abduction', 'knee_flexion', 'ankle_dorsiflexion',
        'pelvis_tilt', 'pelvis_obliquity', 'trunk_flexion', 'trunk_rotation',
        'stride_length', 'step_width', 'cadence', 'grf_vertical', 
        'grf_anterior', 'grf_lateral', 'com_velocity'
    ][:n_features]
    
    # メンタル状態ラベル
    mental_state_labels = [
        'neutral', 'anxious', 'relaxed', 'focused', 'fatigued', 
        'energized', 'distracted'
    ][:n_conditions]
    
    # 時間軸（歩行周期 0-100%）
    t = np.linspace(0, 2 * np.pi, n_timepoints)
    
    # 基本的な歩行パターン（各特徴量の時間的パターン）
    base_patterns = np.zeros((n_features, n_timepoints))
    
    for i in range(n_features):
        # 異なる周波数成分を持つ歩行パターン
        freq = 1 + (i % 3)
        phase = i * np.pi / n_features
        base_patterns[i] = (
            np.sin(freq * t + phase) + 
            0.5 * np.sin(2 * freq * t + phase) +
            0.25 * np.sin(3 * freq * t)
        )
    
    # 正規化
    base_patterns = base_patterns / np.max(np.abs(base_patterns), axis=1, keepdims=True)
    
    # メンタル状態に依存する効果ベクトル
    mental_state_effects = np.random.randn(n_features, n_conditions)
    mental_state_effects = mental_state_effects / np.linalg.norm(mental_state_effects, axis=0)
    
    # 具体的なメンタル状態の効果を設計
    # Anxious: 歩行が速く、ステップ幅が狭い、体幹の動揺が増加
    if n_conditions > 1 and 'anxious' in mental_state_labels:
        idx = mental_state_labels.index('anxious')
        # 歩幅（stride_length）を減少
        if 'stride_length' in feature_labels:
            f_idx = feature_labels.index('stride_length')
            mental_state_effects[f_idx, idx] = -0.8
        # ステップ幅（step_width）を減少
        if 'step_width' in feature_labels:
            f_idx = feature_labels.index('step_width')
            mental_state_effects[f_idx, idx] = -0.5
        # ケイデンスを増加
        if 'cadence' in feature_labels:
            f_idx = feature_labels.index('cadence')
            mental_state_effects[f_idx, idx] = 0.7
    
    # Relaxed: 歩行がゆっくり、自然な動き
    if 'relaxed' in mental_state_labels:
        idx = mental_state_labels.index('relaxed')
        if 'stride_length' in feature_labels:
            f_idx = feature_labels.index('stride_length')
            mental_state_effects[f_idx, idx] = 0.3
        if 'cadence' in feature_labels:
            f_idx = feature_labels.index('cadence')
            mental_state_effects[f_idx, idx] = -0.4
    
    # Fatigued: 全体的に振幅が低下
    if 'fatigued' in mental_state_labels:
        idx = mental_state_labels.index('fatigued')
        mental_state_effects[:, idx] *= 0.7
        if 'stride_length' in feature_labels:
            f_idx = feature_labels.index('stride_length')
            mental_state_effects[f_idx, idx] = -0.6
    
    # 時間依存成分（歩行位相に関連）
    time_effects = base_patterns.copy()
    
    # 相互作用成分（メンタル状態×時間）
    interaction_effects = np.zeros((n_features, n_timepoints, n_conditions))
    
    for c in range(n_conditions):
        # 各条件で異なる時間的修飾
        phase_shift = c * np.pi / (2 * n_conditions)
        amplitude_mod = 1 + 0.3 * np.sin(c * np.pi / n_conditions)
        
        for f in range(n_features):
            interaction_effects[f, :, c] = (
                amplitude_mod * np.sin(t + phase_shift) * 
                mental_state_effects[f, c]
            )
    
    # データを組み立て
    data = np.zeros((n_trials, n_features, n_timepoints, n_conditions))
    
    for trial in range(n_trials):
        # トライアルごとのランダムな変動
        trial_variation = np.random.randn(n_features, 1, 1) * 0.1
        
        for c in range(n_conditions):
            # 時間成分
            data[trial, :, :, c] += time_effect_strength * time_effects
            
            # メンタル状態成分
            data[trial, :, :, c] += mental_state_effect_strength * \
                mental_state_effects[:, c:c+1]
            
            # 相互作用成分
            data[trial, :, :, c] += interaction_strength * \
                interaction_effects[:, :, c]
            
            # トライアル変動
            data[trial, :, :, c] += trial_variation.squeeze()
            
            # ノイズ
            data[trial, :, :, c] += noise_level * \
                np.random.randn(n_features, n_timepoints)
    
    # メタデータ
    metadata = {
        'n_trials': n_trials,
        'n_features': n_features,
        'n_timepoints': n_timepoints,
        'n_conditions': n_conditions,
        'feature_labels': feature_labels,
        'mental_state_labels': mental_state_labels,
        'time_points': np.linspace(0, 100, n_timepoints).tolist(),
        'generation_params': {
            'mental_state_effect_strength': mental_state_effect_strength,
            'time_effect_strength': time_effect_strength,
            'interaction_strength': interaction_strength,
            'noise_level': noise_level,
            'random_state': random_state
        }
    }
    
    return data, metadata


class GaitDataLoader:
    """
    歩行データの読み込みとプリプロセッシング
    
    様々なフォーマットの歩行データを統一的な形式に変換
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Parameters
        ----------
        data_dir : str, optional
            データディレクトリのパス
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self._data: Optional[np.ndarray] = None
        self._metadata: Dict = {}
        
    def load_from_numpy(
        self,
        filepath: str,
        metadata_path: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        NumPy形式のデータを読み込み
        
        Parameters
        ----------
        filepath : str
            .npy または .npz ファイルのパス
        metadata_path : str, optional
            JSONメタデータファイルのパス
            
        Returns
        -------
        data : ndarray
            読み込まれたデータ
        metadata : dict
            メタデータ
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.npz':
            loaded = np.load(filepath, allow_pickle=True)
            self._data = loaded['data']
            if 'metadata' in loaded.files:
                self._metadata = loaded['metadata'].item()
        else:
            self._data = np.load(filepath)
            
        if metadata_path:
            with open(metadata_path, 'r') as f:
                self._metadata = json.load(f)
                
        return self._data, self._metadata
    
    def load_from_csv(
        self,
        filepath: str,
        time_column: str = 'time',
        condition_column: str = 'condition',
        trial_column: str = 'trial',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        CSV形式のデータを読み込み
        
        Parameters
        ----------
        filepath : str
            CSVファイルのパス
        time_column : str
            時間列の名前
        condition_column : str
            条件列の名前
        trial_column : str
            トライアル列の名前
        feature_columns : list of str, optional
            特徴量列のリスト
            
        Returns
        -------
        data : ndarray
            形状 (n_trials, n_features, n_timepoints, n_conditions)
        metadata : dict
            メタデータ
        """
        df = pd.read_csv(filepath)
        
        # 特徴量列を特定
        if feature_columns is None:
            exclude_cols = {time_column, condition_column, trial_column}
            feature_columns = [c for c in df.columns if c not in exclude_cols]
        
        # ユニークな値を取得
        trials = sorted(df[trial_column].unique())
        conditions = sorted(df[condition_column].unique())
        timepoints = sorted(df[time_column].unique())
        
        n_trials = len(trials)
        n_features = len(feature_columns)
        n_timepoints = len(timepoints)
        n_conditions = len(conditions)
        
        # データ配列を作成
        data = np.zeros((n_trials, n_features, n_timepoints, n_conditions))
        
        for t_idx, trial in enumerate(trials):
            for c_idx, cond in enumerate(conditions):
                mask = (df[trial_column] == trial) & (df[condition_column] == cond)
                subset = df.loc[mask].sort_values(time_column)
                
                for f_idx, feature in enumerate(feature_columns):
                    data[t_idx, f_idx, :, c_idx] = subset[feature].values[:n_timepoints]
        
        self._data = data
        self._metadata = {
            'n_trials': n_trials,
            'n_features': n_features,
            'n_timepoints': n_timepoints,
            'n_conditions': n_conditions,
            'feature_labels': feature_columns,
            'mental_state_labels': [str(c) for c in conditions],
            'time_points': list(timepoints)
        }
        
        return self._data, self._metadata
    
    def preprocess(
        self,
        data: np.ndarray,
        normalize: bool = True,
        filter_cutoff: Optional[float] = None,
        resample_points: Optional[int] = None
    ) -> np.ndarray:
        """
        データのプリプロセッシング
        
        Parameters
        ----------
        data : ndarray
            入力データ
        normalize : bool
            正規化するかどうか
        filter_cutoff : float, optional
            ローパスフィルタのカットオフ周波数（0-1）
        resample_points : int, optional
            リサンプリングする時間点数
            
        Returns
        -------
        processed_data : ndarray
            プリプロセス済みデータ
        """
        processed = data.copy()
        
        # リサンプリング
        if resample_points is not None and resample_points != data.shape[2]:
            from scipy.interpolate import interp1d
            
            n_trials, n_features, n_timepoints, n_conditions = data.shape
            new_data = np.zeros((n_trials, n_features, resample_points, n_conditions))
            
            old_t = np.linspace(0, 1, n_timepoints)
            new_t = np.linspace(0, 1, resample_points)
            
            for trial in range(n_trials):
                for feature in range(n_features):
                    for cond in range(n_conditions):
                        interp_func = interp1d(old_t, processed[trial, feature, :, cond])
                        new_data[trial, feature, :, cond] = interp_func(new_t)
            
            processed = new_data
        
        # ローパスフィルタ
        if filter_cutoff is not None:
            from scipy.signal import butter, filtfilt
            
            b, a = butter(4, filter_cutoff, btype='low')
            
            for trial in range(processed.shape[0]):
                for feature in range(processed.shape[1]):
                    for cond in range(processed.shape[3]):
                        processed[trial, feature, :, cond] = filtfilt(
                            b, a, processed[trial, feature, :, cond]
                        )
        
        # 正規化（各特徴量をz-score）
        if normalize:
            mean = processed.mean(axis=(0, 2, 3), keepdims=True)
            std = processed.std(axis=(0, 2, 3), keepdims=True)
            std[std == 0] = 1
            processed = (processed - mean) / std
            
        return processed
    
    def get_trial_averaged_data(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        トライアル平均データを取得
        
        Parameters
        ----------
        data : ndarray, optional
            入力データ。Noneの場合は保存済みデータを使用
            
        Returns
        -------
        averaged : ndarray of shape (n_features, n_timepoints, n_conditions)
        """
        if data is None:
            data = self._data
        
        if data is None:
            raise ValueError("No data available. Load data first.")
            
        return data.mean(axis=0)
    
    def save(
        self,
        data: np.ndarray,
        filepath: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        データを保存
        
        Parameters
        ----------
        data : ndarray
            保存するデータ
        filepath : str
            保存先パス
        metadata : dict, optional
            メタデータ
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.npz':
            if metadata:
                np.savez(filepath, data=data, metadata=metadata)
            else:
                np.savez(filepath, data=data)
        elif filepath.suffix == '.npy':
            np.save(filepath, data)
            if metadata:
                meta_path = filepath.with_suffix('.json')
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @property
    def data(self) -> Optional[np.ndarray]:
        return self._data
    
    @property
    def metadata(self) -> Dict:
        return self._metadata


def create_gait_cycle_time_axis(n_points: int = 100) -> np.ndarray:
    """
    歩行周期の時間軸を作成（0-100% gait cycle）
    
    Parameters
    ----------
    n_points : int
        時間点の数
        
    Returns
    -------
    time_axis : ndarray
        0から100までの時間軸
    """
    return np.linspace(0, 100, n_points)


def align_to_gait_cycle(
    data: np.ndarray,
    heel_strike_indices: List[int],
    target_length: int = 100
) -> np.ndarray:
    """
    データを歩行周期に正規化
    
    Parameters
    ----------
    data : ndarray of shape (n_features, n_samples)
        連続的な歩行データ
    heel_strike_indices : list of int
        ヒールストライクのインデックス
    target_length : int
        正規化後の時間点数
        
    Returns
    -------
    cycles : ndarray of shape (n_cycles, n_features, target_length)
        歩行周期ごとに分割・正規化されたデータ
    """
    from scipy.interpolate import interp1d
    
    n_features = data.shape[0]
    n_cycles = len(heel_strike_indices) - 1
    
    cycles = np.zeros((n_cycles, n_features, target_length))
    
    for i in range(n_cycles):
        start = heel_strike_indices[i]
        end = heel_strike_indices[i + 1]
        cycle_data = data[:, start:end]
        
        # 補間して正規化
        old_t = np.linspace(0, 1, cycle_data.shape[1])
        new_t = np.linspace(0, 1, target_length)
        
        for f in range(n_features):
            interp_func = interp1d(old_t, cycle_data[f])
            cycles[i, f] = interp_func(new_t)
    
    return cycles
