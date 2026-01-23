# MultiVariateMentalDPCA

## Overview

**MultiVariateMentalDPCA** finds associations between **multiple mental variables** and **gait latent dynamics** using Canonical Correlation Analysis (CCA) or Partial Least Squares (PLS).

This is the recommended method when you have:
- Multiple mental scores (wellbeing, anxiety, stress, fatigue)
- Time-series gait data per subject

## Problem Setting

Data structure:
- **Gait**: `[n_subjects, n_timepoints, n_body_factors]`
- **Mental**: `[n_subjects, n_mental_vars]`

Goal: Find which gait features are associated with which mental variables.

```
Gait Data                    Mental Scores
[50, 100, 15]                [50, 4]
 │    │    │                  │   │
 │    │    └─ body factors    │   └─ wellbeing, anxiety, stress, fatigue
 │    └────── timepoints      │
 └─────────── subjects        └────── subjects
```

## Key Insight

Mental scores are time-invariant (constant throughout gait cycle), so we need to:
1. Extract time-invariant gait features (via averaging or latent components)
2. Find associations with mental variables

## Mathematical Formulation

### Step 1: Extract Gait Latent Features

Apply PCA to time-averaged gait features:

$$\tilde{X}[i, n] = \frac{1}{T} \sum_t X_i[n, t]$$

Then PCA:
$$Z = \tilde{X} W_{pca}$$

Where $Z$ is `[n_subjects, n_gait_components]`.

### Step 2: Canonical Correlation Analysis (CCA)

CCA finds linear combinations of gait ($Z$) and mental ($Y$) that are maximally correlated.

Find $a$ and $b$ that maximize:
$$\rho = \text{Corr}(Za, Yb) = \frac{a^T C_{ZY} b}{\sqrt{a^T C_{ZZ} a \cdot b^T C_{YY} b}}$$

Where:
- $C_{ZY}$: Cross-covariance between gait and mental
- $C_{ZZ}$: Gait covariance
- $C_{YY}$: Mental covariance

### Step 3: Feature Associations

Map CCA weights back to original gait features:

$$W_{gait \to mental} = W_{pca} \cdot A \cdot B^T$$

Where $A$ and $B$ are CCA weights.

This gives direct correlations between each gait feature and each mental variable.

## Algorithm

```
Input:
  gait_data[n_subjects, n_timepoints, n_body_factors]
  mental_scores[n_subjects, n_mental_vars]
  n_gait_components

Output:
  feature_correlations[n_body_factors, n_mental_vars]
  canonical_correlations[n_components]

1. Time-average gait data:
   X_avg[n_subjects, n_body_factors] = mean(gait_data, axis=time)

2. Standardize:
   X_avg ← (X_avg - mean) / std
   Y ← (mental_scores - mean) / std

3. PCA on gait features:
   Z = X_avg @ W_pca  # [n_subjects, n_gait_components]

4. CCA between Z and Y:
   a. Compute covariances: C_ZZ, C_YY, C_ZY
   b. Solve: C_ZZ^{-1} C_ZY C_YY^{-1} C_YZ a = λ² a
   c. Get canonical correlations: ρ_k = √λ_k
   d. Get weights: A (gait), B (mental)

5. Compute feature correlations:
   For each gait feature n and mental var m:
     feature_correlations[n, m] = corr(X_avg[:, n], Y[:, m])

6. Return feature_correlations, canonical_correlations
```

## CCA vs PLS Mode

### CCA (Canonical Correlation Analysis)
- Maximizes correlation
- Symmetric (gait ↔ mental)
- Good for understanding relationships

### PLS (Partial Least Squares)
- Maximizes covariance
- Asymmetric (gait → mental)
- Good for prediction

```python
# CCA mode (default)
model = MultiVariateMentalDPCA(method='cca')

# PLS mode
model = MultiVariateMentalDPCA(method='pls')
```

## Output Interpretation

### Feature Correlations

Matrix showing correlation between each gait feature and mental variable:

```
                    Wellbeing  Anxiety  Stress  Fatigue
hip_flexion            0.45    -0.12    -0.08    -0.15
knee_flexion           0.38    -0.05    -0.03    -0.22
stride_length          0.62    -0.31    -0.25    -0.45
trunk_rotation        -0.08     0.52     0.48     0.12
...
```

Interpretation:
- **Positive**: Higher gait value → Higher mental score
- **Negative**: Higher gait value → Lower mental score

### Associations Dictionary

```python
associations = model.get_mental_gait_associations(threshold=0.3)

# Output:
{
    'wellbeing': {
        'stride_length': 0.62,
        'hip_flexion': 0.45,
        'knee_flexion': 0.38
    },
    'anxiety': {
        'trunk_rotation': 0.52
    },
    ...
}
```

Only includes correlations above threshold.

## Code Example

```python
from src.dpca import MultiVariateMentalDPCA
import numpy as np

# Generate data
n_subjects = 50
n_timepoints = 100
n_body_factors = 15
n_mental_vars = 4

gait_data = np.random.randn(n_subjects, n_timepoints, n_body_factors)
mental_scores = np.random.randn(n_subjects, n_mental_vars)

# Define labels
gait_labels = ['hip_flexion', 'hip_abduction', 'knee_flexion', ...]
mental_labels = ['wellbeing', 'anxiety', 'stress', 'fatigue']

# Fit model
model = MultiVariateMentalDPCA(n_gait_components=10, method='cca')
model.fit(
    gait_data, 
    mental_scores,
    gait_labels=gait_labels,
    mental_labels=mental_labels
)

# Get associations
associations = model.get_mental_gait_associations(threshold=0.3)
print("Wellbeing associated with:", associations['wellbeing'])

# Direct correlations
corr_matrix = model.feature_correlations_
print(f"Correlation matrix shape: {corr_matrix.shape}")  # [15, 4]

# Visualize
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 10))
sns.heatmap(
    corr_matrix,
    xticklabels=mental_labels,
    yticklabels=gait_labels,
    cmap='RdBu_r',
    center=0,
    annot=True,
    fmt='.2f'
)
plt.title('Gait-Mental Correlations')
plt.tight_layout()
plt.show()
```

## Canonical Correlations

The `canonical_correlations_` attribute shows the strength of each latent dimension:

```python
print(model.canonical_correlations_)
# [0.85, 0.62, 0.45, 0.23]

# First dimension captures 85% max correlation
# Second dimension captures 62%, etc.
```

High canonical correlations indicate strong gait-mental relationships.

## Practical Considerations

### Sample Size

Rule of thumb: n_subjects > 10 × max(n_gait_components, n_mental_vars)

For 10 gait components and 4 mental variables:
- Minimum: 100 subjects
- Recommended: 200+ subjects

### Multicollinearity

Gait features are often correlated. This is handled by:
1. PCA reduction (n_gait_components < n_body_factors)
2. Regularization in CCA/PLS

### Missing Data

Currently requires complete data. Handle missing values before fitting:
```python
# Remove subjects with missing data
valid_idx = ~np.isnan(mental_scores).any(axis=1)
gait_clean = gait_data[valid_idx]
mental_clean = mental_scores[valid_idx]
```

## Comparison with Other Methods

| Method | Mental Vars | Gait Format | Output |
|--------|-------------|-------------|--------|
| ContinuousScoreDPCA | 1 | [subj, feat, time] | Feature weights |
| MultiVariateMentalDPCA | Multiple | [subj, time, feat] | Correlation matrix |
| DemixedPCA | Categorical | [feat, time, cond] | Variance decomposition |
| DemixedPLS | Continuous | [feat, time, cond] | Predictions |

## Use Cases

1. **Mental health profiling**: Which gait features cluster with anxiety vs. depression?
2. **Intervention assessment**: Do gait patterns change with mental health improvements?
3. **Biomarker discovery**: Identify gait signatures of mental states
4. **Individual differences**: Understand subject-level gait-mental relationships

## Limitations

1. **Cross-sectional only**: Captures between-subject differences, not within-subject changes
2. **Linear associations**: Nonlinear relationships need extensions
3. **Causality**: Cannot infer causal direction (gait→mental or mental→gait)

## References

- Hotelling, H. (1936). Relations between two sets of variates. Biometrika.
- Hardoon, D. R., et al. (2004). Canonical correlation analysis: An overview with application to learning methods. Neural computation.
