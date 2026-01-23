# Gait Latent Dynamics: Mental State Analysis

Analyze the relationship between **gait dynamics** and **mental states** using dimensionality reduction techniques.

## Overview

This project provides tools to find associations between gait patterns and mental variables (wellbeing, anxiety, stress, fatigue, etc.) at the subject level.

### Key Features

- **MultiVariateMentalDPCA**: Find associations between multiple mental variables and gait features using CCA/PLS
- **ContinuousScoreDPCA**: Analyze single continuous mental score vs gait patterns
- **DemixedPCA**: Separate variance into time, condition, and interaction components
- **DemixedPLS**: Supervised version that maximizes covariance with target variables

## Installation

```bash
# Clone repository
git clone https://github.com/Taiyou/gait_latent_dynamics.git
cd gait_latent_dynamics

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Subject-Level Analysis (Recommended)

```python
from src.dpca import MultiVariateMentalDPCA
import numpy as np

# Data structure
# gait_data: [n_subjects, n_timepoints, n_body_factors]
# mental_scores: [n_subjects, n_mental_vars]

gait_data = np.random.randn(50, 100, 15)  # 50 subjects, 100 timepoints, 15 body factors
mental_scores = np.random.randn(50, 4)    # 50 subjects, 4 mental variables

# Fit model
model = MultiVariateMentalDPCA(n_gait_components=10, method='cca')
model.fit(
    gait_data, 
    mental_scores,
    gait_labels=['hip_flexion', 'knee_flexion', ...],
    mental_labels=['wellbeing', 'anxiety', 'stress', 'fatigue']
)

# Get associations
associations = model.get_mental_gait_associations()

# Correlation heatmap
correlations = model.feature_correlations_  # [n_body_factors, n_mental_vars]
```

### Single Mental Variable

```python
from src.dpca import ContinuousScoreDPCA

# gait_data: [n_subjects, n_features, n_timepoints]
# wellbeing_scores: [n_subjects]

model = ContinuousScoreDPCA(n_components=5)
model.fit(gait_data, wellbeing_scores)

# Predict wellbeing from gait
predicted = model.predict_score(new_gait_data)

# Get feature importance
summary = model.summary()
```

## Data Structure

### Subject-Level Analysis

| Data | Shape | Description |
|------|-------|-------------|
| Gait Dynamics | `[n_subjects, n_timepoints, n_body_factors]` | Time-varying gait patterns per subject |
| Mental Scores | `[n_subjects, n_mental_vars]` | Time-invariant mental scores per subject |

### Condition-Based Analysis (dPCA)

| Data | Shape | Description |
|------|-------|-------------|
| Gait Data | `[n_features, n_timepoints, n_conditions]` | Trial-averaged gait by condition |

## Project Structure

```
gait_latent_dynamics/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dpca.py           # Core implementations
│   ├── data_loader.py    # Data loading/generation
│   └── visualization.py  # Visualization tools
├── examples/
│   └── demo_dpca_analysis.py
├── notebooks/
│   ├── gait_mental_analysis.ipynb    # Subject-level analysis
│   ├── demixed_pca_explained.ipynb   # dPCA theory
│   └── dpca_tutorial.ipynb           # Tutorial
├── tests/
│   └── test_dpca.py      # Test suite
└── outputs/
```

## Available Classes

### `MultiVariateMentalDPCA`

Find associations between multiple mental variables and gait latent dynamics.

```python
model = MultiVariateMentalDPCA(
    n_gait_components=10,   # Latent gait components to extract
    method='cca'            # 'cca' or 'pls'
)

model.fit(gait_data, mental_scores, gait_labels=..., mental_labels=...)

# Get associations (threshold for significance)
associations = model.get_mental_gait_associations(threshold=0.3)

# Direct feature correlations
model.feature_correlations_  # [n_body_factors, n_mental_vars]
```

### `ContinuousScoreDPCA`

For single continuous mental score (e.g., wellbeing only).

```python
model = ContinuousScoreDPCA(n_components=5)
model.fit(gait_data, scores, feature_labels=...)

# Predict scores from gait
predicted = model.predict_score(new_data)

# Feature weights
model.score_weights_  # [n_features]
```

### `DemixedPCA`

Separate variance into time, condition, and interaction components.

```python
dpca = DemixedPCA(n_components=10, regularizer='auto')
dpca.fit(data, feature_labels=...)

# Transform
Z = dpca.transform(data)  # All marginalizations
Z_cond = dpca.transform(data, marginalization='condition')

# Variance decomposition
summary = dpca.get_demixing_summary()
```

### `DemixedPLS`

Supervised demixed analysis (maximizes covariance with Y).

```python
dpls = DemixedPLS(n_components=5)
dpls.fit(X, Y, feature_labels=..., condition_labels=...)

# Predict mental scores
Y_pred = dpls.predict(X_new)

# Feature importance
importance = dpls.get_feature_importance('condition')
```

## Gait Features

| Feature | Description |
|---------|-------------|
| hip_flexion | Hip joint flexion angle |
| hip_abduction | Hip joint abduction angle |
| knee_flexion | Knee joint flexion angle |
| ankle_dorsiflexion | Ankle dorsiflexion angle |
| pelvis_tilt | Pelvis anterior/posterior tilt |
| pelvis_obliquity | Pelvis lateral tilt |
| trunk_flexion | Trunk forward/backward lean |
| trunk_rotation | Trunk rotation |
| stride_length | Step length |
| step_width | Lateral distance between feet |
| cadence | Steps per minute |
| grf_vertical | Vertical ground reaction force |
| grf_anterior | Anterior-posterior GRF |
| grf_lateral | Lateral GRF |
| com_velocity | Center of mass velocity |

## Mental Variables

| Variable | Description |
|----------|-------------|
| wellbeing | Overall mental wellbeing (positive) |
| anxiety | Anxiety level |
| stress | Stress level |
| fatigue | Fatigue/tiredness level |

## Running Tests

```bash
# Run all tests
python -m pytest tests/test_dpca.py -v

# Quick test
python tests/test_dpca.py
```

## Running Demos

```bash
# Command line demo
python examples/demo_dpca_analysis.py

# Jupyter notebooks
jupyter notebook notebooks/gait_mental_analysis.ipynb
```

## Documentation

Detailed explanations of each method:

| Method | Description | Documentation |
|--------|-------------|---------------|
| DemixedPCA | Separate variance by source | [docs/demixed_pca.md](docs/demixed_pca.md) |
| DemixedPLS | Supervised prediction | [docs/demixed_pls.md](docs/demixed_pls.md) |
| ContinuousScoreDPCA | Single mental score | [docs/continuous_score_dpca.md](docs/continuous_score_dpca.md) |
| MultiVariateMentalDPCA | Multiple mental variables | [docs/multivariate_mental_dpca.md](docs/multivariate_mental_dpca.md) |

## References

- Kobak, D., et al. (2016). Demixed principal component analysis of neural population data. eLife, 5, e10989.
- Hotelling, H. (1936). Relations between two sets of variates. Biometrika.
- Wold, H. (1985). Partial least squares. Encyclopedia of statistical sciences.

## License

MIT License
