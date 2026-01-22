# Gait Latent Dynamics: Demixed PCA Analysis

歩行ダイナミクスとメンタル状態の対応関係を分析するためのDemixed PCA (dPCA) 実装

## 概要

このプロジェクトは、歩行運動データからメンタル状態（不安、リラックス、集中など）に関連する成分を分離・抽出するためのdemixed PCA手法を提供します。

### Demixed PCA とは？

標準的なPCAはデータの分散を最大化する方向を見つけますが、各成分が何を表しているかの解釈が困難です。dPCAは、データの分散を以下のような異なる「ソース」に分解します：

- **時間成分 (Time)**: 歩行周期に沿った変動パターン
- **条件成分 (Condition)**: メンタル状態による変動
- **相互作用成分 (Interaction)**: 時間と条件の相互作用

## インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd gait_latent_dynamics

# 依存関係をインストール
pip install -r requirements.txt
```

## クイックスタート

```python
from src.dpca import DemixedPCA
from src.data_loader import generate_synthetic_gait_data, GaitDataLoader
from src.visualization import DPCAVisualizer

# 1. データの生成（または読み込み）
data, metadata = generate_synthetic_gait_data(
    n_trials=50,
    n_features=15,
    n_timepoints=100,
    n_conditions=5,
    random_state=42
)

# 2. 前処理
loader = GaitDataLoader()
data_processed = loader.preprocess(data, normalize=True)
data_averaged = loader.get_trial_averaged_data(data_processed)

# 3. dPCAフィッティング
dpca = DemixedPCA(n_components=10, regularizer='auto')
dpca.fit(data_averaged, feature_labels=metadata['feature_labels'])

# 4. 結果の分析
summary = dpca.get_demixing_summary()
print("Variance Decomposition:", summary)

# 5. 可視化
viz = DPCAVisualizer()
viz.plot_comprehensive_summary(
    dpca,
    data_averaged,
    metadata['mental_state_labels']
)
```

## プロジェクト構造

```
gait_latent_dynamics/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dpca.py           # dPCAコア実装
│   ├── data_loader.py    # データ読み込み/生成
│   └── visualization.py  # 可視化ツール
├── examples/
│   └── demo_dpca_analysis.py
├── notebooks/
│   └── dpca_tutorial.ipynb
└── outputs/              # 出力ファイル保存先
```

## 主要なクラスと関数

### `DemixedPCA`

dPCAのメインクラス。

```python
dpca = DemixedPCA(
    n_components=10,      # 各marginalizationの成分数
    regularizer='auto',   # 正則化パラメータ（'auto'で自動決定）
    n_splits=5            # 交差検証の分割数
)

# フィット
dpca.fit(X, feature_labels=['hip', 'knee', ...])

# 変換
transformed = dpca.transform(X)  # 全marginalization
time_comp = dpca.transform(X, marginalization='time')  # 特定のみ

# 逆変換（再構成）
reconstructed = dpca.inverse_transform(transformed)

# サマリー
summary = dpca.get_demixing_summary()
```

### `GaitDPCA`

歩行解析に特化した拡張クラス。

```python
gait_dpca = GaitDPCA(n_components=10)
gait_dpca.fit_with_labels(
    X,
    mental_state_labels=['neutral', 'anxious', 'relaxed'],
    feature_labels=['hip_flexion', 'knee_flexion', ...]
)

# メンタル状態間の分離度を分析
separation = gait_dpca.analyze_mental_state_separation(X, mental_state_labels)
```

### `GaitDataLoader`

データの読み込みと前処理。

```python
loader = GaitDataLoader()

# NumPyファイルから読み込み
data, metadata = loader.load_from_numpy('gait_data.npz')

# CSVから読み込み
data, metadata = loader.load_from_csv(
    'gait_data.csv',
    time_column='time',
    condition_column='mental_state'
)

# 前処理
data_processed = loader.preprocess(
    data,
    normalize=True,
    filter_cutoff=0.8,
    resample_points=100
)
```

### `DPCAVisualizer`

結果の可視化。

```python
viz = DPCAVisualizer()

# 説明分散比
viz.plot_explained_variance(dpca)

# 成分の時系列
viz.plot_component_timecourse(dpca, X, marginalization='condition', components=[0, 1, 2])

# メンタル状態の分離
viz.plot_mental_state_separation(dpca, X, mental_state_labels)

# 成分重み
viz.plot_component_weights(dpca, marginalization='condition', component=0)

# 包括的サマリー
viz.plot_comprehensive_summary(dpca, X, mental_state_labels, feature_labels)
```

## データ形式

入力データは以下の形状を期待します：

- **4D**: `(n_trials, n_features, n_timepoints, n_conditions)` - トライアルレベルデータ
- **3D**: `(n_features, n_timepoints, n_conditions)` - トライアル平均データ

### 例：

```python
# 50トライアル、15特徴量、100時間点、5条件
data.shape = (50, 15, 100, 5)

# 特徴量: 関節角度、歩行パラメータ、地面反力など
# 時間: 歩行周期 0-100%
# 条件: メンタル状態（neutral, anxious, relaxed, focused, fatigued）
```

## 合成データ生成

```python
from src.data_loader import generate_synthetic_gait_data

data, metadata = generate_synthetic_gait_data(
    n_trials=50,
    n_features=15,
    n_timepoints=100,
    n_conditions=5,
    mental_state_effect_strength=0.3,  # メンタル状態の影響
    time_effect_strength=0.5,          # 時間の影響
    interaction_strength=0.15,         # 相互作用
    noise_level=0.1,
    random_state=42
)
```

## デモの実行

```bash
# コマンドラインから
python examples/demo_dpca_analysis.py

# Jupyterノートブック
jupyter notebook notebooks/dpca_tutorial.ipynb
```

## 参考文献

- Kobak, D., Brendel, W., Constantinidis, C., Feierstein, C. E., Kepecs, A., Mainen, Z. F., ... & Machens, C. K. (2016). Demixed principal component analysis of neural population data. eLife, 5, e10989.

## ライセンス

MIT License
