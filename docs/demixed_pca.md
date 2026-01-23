# Demixed PCA (dPCA)

## Overview

**Demixed Principal Component Analysis (dPCA)** is a linear dimensionality reduction technique that separates the variance in data into interpretable components based on task parameters.

Unlike standard PCA which finds directions of maximum variance, dPCA finds directions that capture variance attributable to specific experimental factors (e.g., time, condition, their interaction).

## Problem Setting

Consider neural/behavioral data with structure:
- **Features** (N): e.g., joint angles, muscle activities
- **Time** (T): e.g., gait cycle 0-100%
- **Conditions** (C): e.g., mental states (neutral, anxious, relaxed)

Data shape: `X[n_features, n_timepoints, n_conditions]`

## Mathematical Formulation

### 1. Marginalization

The key insight of dPCA is to decompose the data into **marginalizations** - averages over different subsets of parameters.

For data `X[n, t, c]`:

**Time marginalization** (average over conditions):
$$X_t[n, t] = \frac{1}{C} \sum_c X[n, t, c]$$

**Condition marginalization** (average over time):
$$X_c[n, c] = \frac{1}{T} \sum_t X[n, t, c]$$

**Interaction** (residual):
$$X_{tc}[n, t, c] = X[n, t, c] - X_t[n, t] - X_c[n, c] + \bar{X}$$

### 2. Covariance Decomposition

Total covariance can be decomposed:
$$C_{total} = C_{time} + C_{condition} + C_{interaction} + C_{noise}$$

Where:
- $C_{time}$: Covariance from time-varying patterns
- $C_{condition}$: Covariance from condition differences
- $C_{interaction}$: Covariance from time×condition interaction

### 3. Optimization

For each marginalization φ ∈ {time, condition, interaction}, dPCA finds:

**Encoder** $F_φ$ and **Decoder** $D_φ$ that minimize:

$$\mathcal{L}_φ = ||X_φ - D_φ F_φ X||^2 + λ||D_φ F_φ||^2$$

Subject to:
- $F_φ D_φ = I$ (orthogonality)
- Components ordered by explained variance

### 4. Solution

The optimal encoder/decoder are found by solving a generalized eigenvalue problem:

$$C_φ D_φ = C_{total} D_φ Λ$$

Where $Λ$ is a diagonal matrix of eigenvalues (explained variance).

## Algorithm

```
Input: X[n_features, n_timepoints, n_conditions]
Output: Encoders F_φ, Decoders D_φ for each marginalization

1. Center data: X ← X - mean(X)

2. Compute marginalizations:
   X_time ← mean(X, axis=conditions)
   X_cond ← mean(X, axis=time)
   X_inter ← X - X_time - X_cond + global_mean

3. For each marginalization φ:
   a. Compute covariance: C_φ = X_φ @ X_φ.T
   b. Compute total covariance: C_total = X @ X.T
   c. Add regularization: C_total ← C_total + λI
   d. Solve: C_φ D = C_total D Λ
   e. Select top k components by eigenvalue
   f. Compute encoder: F_φ = D_φ^{-1}

4. Return {F_φ, D_φ} for φ ∈ {time, condition, interaction}
```

## Interpretation

### Variance Decomposition

After fitting, you can see how much variance is explained by each source:

| Source | Interpretation |
|--------|----------------|
| Time | Patterns that vary across gait cycle (common to all conditions) |
| Condition | Differences between mental states (constant across time) |
| Interaction | Patterns where mental state effect changes over time |

### Component Time Courses

Transform data to see how each component evolves:

```python
Z_time = dpca.transform(X, marginalization='time')
# Z_time[component, timepoint] - time-varying pattern

Z_cond = dpca.transform(X, marginalization='condition')  
# Z_cond[component, timepoint, condition] - condition-specific patterns
```

## Comparison with Standard PCA

| Aspect | PCA | dPCA |
|--------|-----|------|
| Objective | Max variance | Max variance per source |
| Components | Mixed | Demixed by source |
| Interpretability | Low | High |
| Supervision | None | Requires condition labels |

### Visual Example

```
Standard PCA Component:
  ┌─────────────────────────┐
  │ Time pattern +          │  ← Mixed, hard to interpret
  │ Condition effect +      │
  │ Interaction            │
  └─────────────────────────┘

dPCA Components:
  Time Component:           Condition Component:
  ┌───────────────┐        ┌───────────────┐
  │ Time pattern  │        │ Cond. effect  │  ← Clean separation
  └───────────────┘        └───────────────┘
```

## Use Cases

1. **Neuroscience**: Separate stimulus-driven vs. choice-related neural activity
2. **Gait Analysis**: Separate gait cycle patterns from mental state effects
3. **EEG/EMG**: Separate temporal dynamics from task conditions

## Regularization

The regularization parameter λ controls the trade-off:
- **λ = 0**: Pure demixing (may be unstable)
- **λ large**: More like standard PCA (stable but less demixed)

`regularizer='auto'` uses cross-validation to find optimal λ.

## Code Example

```python
from src.dpca import DemixedPCA

# Data: [15 features, 100 timepoints, 3 conditions]
X = np.random.randn(15, 100, 3)

# Fit dPCA
dpca = DemixedPCA(n_components=5, regularizer='auto')
dpca.fit(X, feature_labels=['hip', 'knee', ...])

# Get variance decomposition
summary = dpca.get_demixing_summary()
# {'time': 0.45, 'condition': 0.30, 'interaction': 0.15, 'noise': 0.10}

# Transform
Z_cond = dpca.transform(X, marginalization='condition')

# Reconstruct
X_reconstructed = dpca.inverse_transform({'condition': Z_cond})
```

## References

- Kobak, D., et al. (2016). Demixed principal component analysis of neural population data. eLife, 5, e10989.
