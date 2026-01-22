"""
Gait Latent Dynamics - Demixed PCA Analysis

歩行ダイナミクスとメンタル状態の対応関係を分析するためのdemixed PCA実装
"""

from .dpca import DemixedPCA, GaitDPCA, ContinuousScoreDPCA
from .data_loader import GaitDataLoader, generate_synthetic_gait_data
from .visualization import DPCAVisualizer

__version__ = "0.1.0"
__all__ = [
    "DemixedPCA", 
    "GaitDPCA", 
    "ContinuousScoreDPCA",
    "GaitDataLoader", 
    "generate_synthetic_gait_data", 
    "DPCAVisualizer"
]
