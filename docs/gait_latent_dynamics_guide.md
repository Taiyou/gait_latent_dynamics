# Gait Latent Dynamics Analysis Guide

A practical guide for analyzing the relationship between **gait dynamics** and **mental states** using dPCA and dPLS.

## Table of Contents

1. [Introduction](#introduction)
2. [Gait Data Overview](#gait-data-overview)
3. [Mental State Variables](#mental-state-variables)
4. [Analysis Workflow](#analysis-workflow)
5. [Method Selection](#method-selection)
6. [Practical Examples](#practical-examples)
7. [Interpretation Guidelines](#interpretation-guidelines)

---

## Introduction

### What is Gait Latent Dynamics?

Gait (walking) patterns contain high-dimensional information about a person's physical and mental state. **Latent dynamics** refers to the underlying low-dimensional structure that explains the observed gait variations.

```
Observed Gait Data               Latent Dynamics
┌─────────────────────┐         ┌─────────────┐
│ 15 joint angles     │         │ 3-5 latent  │
│ × 100 timepoints    │  ───►   │ components  │
│ = 1500 dimensions   │         │             │
└─────────────────────┘         └─────────────┘
```

### Why Analyze Gait-Mental Relationships?

Research shows that mental states affect gait patterns:

| Mental State | Gait Effect |
|--------------|-------------|
| Anxiety | Shorter stride, increased trunk stiffness |
| Depression | Slower speed, reduced arm swing |
| Fatigue | Irregular rhythm, reduced push-off |
| High Wellbeing | Longer stride, smoother patterns |

---

## Gait Data Overview

### Common Gait Features

| Category | Features | Description |
|----------|----------|-------------|
| **Joint Angles** | hip_flexion, knee_flexion, ankle_dorsiflexion | Angular positions of lower limb joints |
| **Pelvis/Trunk** | pelvis_tilt, trunk_rotation | Core stability indicators |
| **Spatial** | stride_length, step_width | Step geometry |
| **Temporal** | cadence | Rhythm (steps/minute) |
| **Kinetic** | grf_vertical, grf_anterior | Ground reaction forces |
| **Derived** | com_velocity | Center of mass movement |

### Gait Cycle

One complete gait cycle (stride) = 0-100%:

```
0%              50%              100%
│───────────────│────────────────│
Heel    Toe    Heel    Toe     Heel
Strike  Off    Strike  Off     Strike
        (swing phase)

Right Leg: ████████░░░░░░░░████████
Left Leg:  ░░░░░░░░████████░░░░░░░░
           stance   swing    stance
```

### Data Shapes

**Subject-level analysis** (recommended):
```python
gait_data.shape = (n_subjects, n_timepoints, n_body_factors)
# Example: (50, 100, 15) = 50 subjects, 100% gait cycle, 15 features

mental_scores.shape = (n_subjects, n_mental_vars)
# Example: (50, 4) = 50 subjects, 4 mental variables
```

**Condition-based analysis** (traditional dPCA):
```python
gait_data.shape = (n_features, n_timepoints, n_conditions)
# Example: (15, 100, 3) = 15 features, 100% gait cycle, 3 conditions
```

---

## Mental State Variables

### Time-Invariant Nature

Mental scores are **constant throughout a gait cycle**:

```
Subject 1 (Wellbeing = 0.8):
  t=0%   t=25%  t=50%  t=75%  t=100%
    │      │      │      │      │
    └──────┴──────┴──────┴──────┘
           All = 0.8

Subject 2 (Wellbeing = 0.3):
  t=0%   t=25%  t=50%  t=75%  t=100%
    │      │      │      │      │
    └──────┴──────┴──────┴──────┘
           All = 0.3
```

### Common Mental Variables

| Variable | Scale | Description |
|----------|-------|-------------|
| Wellbeing | 0-100 | Overall mental wellness (positive) |
| Anxiety | 0-10 | State anxiety level |
| Stress | 0-10 | Perceived stress |
| Fatigue | 0-10 | Physical/mental tiredness |
| Depression | 0-27 (PHQ-9) | Depression symptoms |

### Expected Gait-Mental Associations

Based on literature:

```
Wellbeing ↑  →  Stride length ↑, COM velocity ↑, Arm swing ↑
Anxiety ↑   →  Trunk stiffness ↑, Step width ↑, Stride length ↓
Fatigue ↑   →  Cadence ↓, GRF ↓, Variability ↑
Depression ↑ →  Gait speed ↓, Posture ↓, Rhythm irregularity ↑
```

---

## Analysis Workflow

### Step 1: Data Preparation

```python
import numpy as np
from src.dpca import MultiVariateMentalDPCA, DemixedPCA, DemixedPLS

# Load your gait data
# gait_data: [n_subjects, n_timepoints, n_body_factors]
# mental_scores: [n_subjects, n_mental_vars]

# Define labels
gait_labels = [
    'hip_flexion', 'hip_abduction', 'knee_flexion', 'ankle_dorsiflexion',
    'pelvis_tilt', 'pelvis_obliquity', 'trunk_flexion', 'trunk_rotation',
    'stride_length', 'step_width', 'cadence',
    'grf_vertical', 'grf_anterior', 'grf_lateral', 'com_velocity'
]

mental_labels = ['wellbeing', 'anxiety', 'stress', 'fatigue']
```

### Step 2: Exploratory Analysis

```python
# Check data distributions
print(f"Gait data shape: {gait_data.shape}")
print(f"Mental scores shape: {mental_scores.shape}")

# Check for missing values
print(f"Missing gait: {np.isnan(gait_data).sum()}")
print(f"Missing mental: {np.isnan(mental_scores).sum()}")

# Visualize mental score distributions
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(mental_labels), figsize=(12, 3))
for i, label in enumerate(mental_labels):
    axes[i].hist(mental_scores[:, i], bins=20)
    axes[i].set_title(label)
plt.tight_layout()
```

### Step 3: Choose Analysis Method

```
┌─────────────────────────────────────────────────────────────┐
│                    METHOD SELECTION                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Mental scores = continuous (0-100)?                        │
│  ├── YES → How many variables?                              │
│  │         ├── 1 variable → ContinuousScoreDPCA            │
│  │         └── Multiple → MultiVariateMentalDPCA           │
│  │                                                          │
│  └── NO (categorical: anxious/neutral/relaxed)             │
│           ├── Understand structure → DemixedPCA            │
│           └── Predict outcomes → DemixedPLS                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 4: Fit Model

```python
# For multiple continuous mental scores (recommended)
model = MultiVariateMentalDPCA(n_gait_components=10, method='cca')
model.fit(gait_data, mental_scores, 
          gait_labels=gait_labels, 
          mental_labels=mental_labels)

# Get results
associations = model.get_mental_gait_associations(threshold=0.3)
correlations = model.feature_correlations_
```

### Step 5: Visualize & Interpret

```python
import seaborn as sns

# Heatmap of correlations
plt.figure(figsize=(8, 10))
sns.heatmap(
    correlations,
    xticklabels=mental_labels,
    yticklabels=gait_labels,
    cmap='RdBu_r',
    center=0,
    annot=True,
    fmt='.2f',
    vmin=-1, vmax=1
)
plt.title('Gait-Mental Correlations')
plt.tight_layout()
plt.show()
```

---

## Method Selection

### When to Use Each Method

| Scenario | Best Method | Why |
|----------|-------------|-----|
| Single wellbeing score per subject | ContinuousScoreDPCA | Optimized for univariate |
| Multiple mental scores (wellbeing, anxiety, stress) | MultiVariateMentalDPCA | Handles multivariate |
| Categorical conditions (anxious vs. relaxed) | DemixedPCA | Designed for conditions |
| Predict mental state from gait | DemixedPLS | Supervised prediction |
| Understand variance sources | DemixedPCA | Variance decomposition |

### Data Format Requirements

| Method | Gait Shape | Mental Shape |
|--------|-----------|--------------|
| MultiVariateMentalDPCA | `[subjects, time, features]` | `[subjects, vars]` |
| ContinuousScoreDPCA | `[subjects, features, time]` | `[subjects]` |
| DemixedPCA | `[features, time, conditions]` | Labels only |
| DemixedPLS | `[features, time, conditions]` | `[conditions]` or `[conditions, vars]` |

---

## Practical Examples

### Example 1: Subject-Level Analysis with Multiple Mental Scores

**Goal**: Find which gait features are associated with wellbeing, anxiety, and fatigue.

```python
from src.dpca import MultiVariateMentalDPCA
import numpy as np

# Data
n_subjects = 100
n_timepoints = 100
n_features = 15
n_mental = 3

gait_data = np.random.randn(n_subjects, n_timepoints, n_features)
mental_scores = np.random.randn(n_subjects, n_mental)

gait_labels = ['hip_flex', 'knee_flex', 'ankle_flex', 'pelvis_tilt', 
               'trunk_rot', 'stride_len', 'step_width', 'cadence',
               'grf_vert', 'grf_ant', 'grf_lat', 'com_vel',
               'hip_abd', 'pelvis_obl', 'trunk_flex']
mental_labels = ['wellbeing', 'anxiety', 'fatigue']

# Fit model
model = MultiVariateMentalDPCA(n_gait_components=8, method='cca')
model.fit(gait_data, mental_scores, 
          gait_labels=gait_labels, 
          mental_labels=mental_labels)

# Results
print("=== Canonical Correlations ===")
for i, cc in enumerate(model.canonical_correlations_):
    print(f"  CC{i+1}: {cc:.3f}")

print("\n=== Top Associations ===")
associations = model.get_mental_gait_associations(threshold=0.25)
for mental_var, gait_assoc in associations.items():
    print(f"\n{mental_var.upper()}:")
    for feat, corr in sorted(gait_assoc.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat}: r={corr:+.3f}")
```

### Example 2: Condition-Based Analysis with dPCA

**Goal**: Separate time-varying patterns from condition (mental state) effects.

```python
from src.dpca import DemixedPCA
from src.data_loader import generate_synthetic_gait_data

# Generate condition-based data
data, metadata = generate_synthetic_gait_data(
    n_trials=50,
    n_features=15,
    n_timepoints=100,
    n_conditions=3,  # neutral, anxious, relaxed
    mental_state_effect_strength=0.4,
    time_effect_strength=0.5,
    interaction_strength=0.2,
    random_state=42
)

# Average over trials
data_avg = data.mean(axis=0)  # [features, time, conditions]

# Fit dPCA
dpca = DemixedPCA(n_components=5, regularizer='auto')
dpca.fit(data_avg, feature_labels=metadata['feature_labels'])

# Variance decomposition
summary = dpca.get_demixing_summary()
print("=== Variance Decomposition ===")
for source, ratio in summary.items():
    bar = "█" * int(ratio * 50)
    print(f"  {source:12s}: {ratio:.1%} {bar}")

# Transform to see condition differences
Z_cond = dpca.transform(data_avg, marginalization='condition')
print(f"\nCondition components shape: {Z_cond.shape}")
# [n_components, n_timepoints, n_conditions]
```

### Example 3: Predicting Mental Scores with dPLS

**Goal**: Build a model to predict mental scores from gait patterns.

```python
from src.dpca import DemixedPLS
import numpy as np

# Condition-based data
X = np.random.randn(15, 100, 5)  # [features, time, conditions]
Y = np.random.randn(5, 2)        # [conditions, mental_vars]

feature_labels = ['hip', 'knee', 'ankle', 'pelvis', 'trunk',
                  'stride', 'width', 'cadence', 'grf_v', 'grf_a',
                  'grf_l', 'com', 'abd', 'obl', 'flex']
condition_labels = ['low_wellbeing', 'mid_low', 'mid', 'mid_high', 'high_wellbeing']

# Fit dPLS
dpls = DemixedPLS(n_components=3, regularizer='auto')
dpls.fit(X, Y, feature_labels=feature_labels, condition_labels=condition_labels)

# Predict
Y_pred = dpls.predict(X, marginalization='condition')
print(f"Predicted shape: {Y_pred.shape}")

# Feature importance
importance = dpls.get_feature_importance('condition')
print("\n=== Feature Importance for Prediction ===")
for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
    bar = "█" * int(imp * 30)
    print(f"  {feat:10s}: {imp:.3f} {bar}")
```

---

## Interpretation Guidelines

### Reading Correlation Heatmaps

```
                 wellbeing  anxiety  fatigue
hip_flexion        +0.45    -0.12    -0.18
stride_length      +0.62    -0.35    -0.42
trunk_rotation     -0.08    +0.48    +0.15
```

**Interpretation**:
- **Stride length +0.62 with wellbeing**: Longer strides → Higher wellbeing
- **Stride length -0.42 with fatigue**: Longer strides → Lower fatigue
- **Trunk rotation +0.48 with anxiety**: More rotation → Higher anxiety (compensatory stability)

### Variance Decomposition (dPCA)

```
Time:        45%  ████████████████████████
Condition:   30%  ████████████████
Interaction: 15%  ████████
Noise:       10%  █████
```

**Interpretation**:
- **Time (45%)**: Largest source = gait cycle patterns (normal)
- **Condition (30%)**: Mental state explains 30% of variance
- **Interaction (15%)**: Mental state effect changes across gait cycle

### Feature Importance (dPLS)

```
stride_length: 0.35 ██████████
com_velocity:  0.28 ████████
hip_flexion:   0.22 ██████
```

**Interpretation**:
These features are most predictive of mental scores. Focus clinical attention on these markers.

### Practical Thresholds

| Correlation | Interpretation |
|-------------|----------------|
| |r| < 0.2 | Negligible |
| 0.2 ≤ |r| < 0.4 | Weak |
| 0.4 ≤ |r| < 0.6 | Moderate |
| 0.6 ≤ |r| < 0.8 | Strong |
| |r| ≥ 0.8 | Very strong |

---

## Best Practices

### 1. Sample Size

- **Minimum**: 10 subjects per variable
- **Recommended**: 50+ subjects for stable estimates
- **Ideal**: 100+ subjects for robust conclusions

### 2. Data Quality

```python
# Check for outliers
from scipy import stats
z_scores = np.abs(stats.zscore(gait_data, axis=0))
outliers = (z_scores > 3).any(axis=(1, 2))
print(f"Potential outliers: {outliers.sum()} subjects")

# Consider removing or investigating
gait_clean = gait_data[~outliers]
```

### 3. Normalization

```python
# Standardize gait features (recommended)
from sklearn.preprocessing import StandardScaler

# Reshape for scaling
gait_2d = gait_data.reshape(n_subjects, -1)
scaler = StandardScaler()
gait_scaled = scaler.fit_transform(gait_2d)
gait_data = gait_scaled.reshape(n_subjects, n_timepoints, n_features)
```

### 4. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# For prediction tasks, use cross-validation
# to estimate generalization performance
```

### 5. Multiple Comparisons

When testing many gait-mental associations, correct for multiple comparisons:

```python
from scipy.stats import false_discovery_control

# Get p-values for each correlation
# Apply FDR correction
significant = false_discovery_control(p_values, method='bh') < 0.05
```

---

## Summary

| Goal | Method | Key Output |
|------|--------|------------|
| Find gait-mental associations | MultiVariateMentalDPCA | Correlation matrix |
| Understand variance sources | DemixedPCA | Decomposition summary |
| Predict mental scores | DemixedPLS | Predictions + importance |
| Single mental variable | ContinuousScoreDPCA | Feature weights |

**Recommended workflow**:
1. Start with `MultiVariateMentalDPCA` for exploration
2. Use heatmap to identify key associations
3. If prediction needed, use `DemixedPLS` for refinement
4. Validate findings with cross-validation
