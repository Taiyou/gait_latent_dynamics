# Demixed PLS (dPLS)

## Overview

**Demixed Partial Least Squares (dPLS)** extends dPCA by incorporating supervision. Instead of just decomposing variance, it finds components that maximally predict a target variable (e.g., mental scores).

## Motivation

dPCA limitation: Components capture variance but not necessarily predictive information.

**Example**: A gait feature with high variance might not predict mental state at all.

dPLS solution: Find components that maximize covariance with the target.

## Problem Setting

- **X**: Gait data `[n_features, n_timepoints, n_conditions]`
- **Y**: Mental scores `[n_conditions]` or `[n_conditions, n_targets]`

Goal: Find latent components of X that best predict Y.

## Mathematical Formulation

### Standard PLS

PLS finds weight vectors w and c that maximize:

$$\max_{w, c} \text{Cov}(Xw, Yc) = w^T X^T Y c$$

Subject to: $||w|| = 1$, $||c|| = 1$

### Demixed PLS

dPLS applies marginalization before PLS:

1. **Marginalize X** by desired source (time, condition, interaction)
2. **Fit PLS** on marginalized data
3. **Store** separate encoders for each marginalization

For condition marginalization:
$$X_{cond}[n, c] = \frac{1}{T} \sum_t X[n, t, c]$$

Then fit PLS on $X_{cond}$ vs $Y$.

## Algorithm: NIPALS

We use the NIPALS (Nonlinear Iterative Partial Least Squares) algorithm:

```
Input: X[n_samples, n_features], Y[n_samples, n_targets], n_components
Output: Weights W, Loadings P, Y_loadings Q

1. For each component k = 1, ..., n_components:
   
   a. Initialize u = Y[:, 0]  # First column of Y
   
   b. Iterate until convergence:
      - w = X.T @ u / (u.T @ u)      # X weight
      - w = w / ||w||                 # Normalize
      - t = X @ w                     # X score
      - c = Y.T @ t / (t.T @ t)      # Y weight
      - c = c / ||c||                 # Normalize
      - u_new = Y @ c                 # Y score
      - Check convergence: ||u_new - u|| < ε
      - u = u_new
   
   c. Compute loadings:
      - p = X.T @ t / (t.T @ t)      # X loading
      - q = Y.T @ t / (t.T @ t)      # Y loading
   
   d. Deflate:
      - X = X - t @ p.T              # Remove component from X
      - Y = Y - t @ q.T              # Remove component from Y
   
   e. Store: W[:, k] = w, P[:, k] = p, Q[:, k] = q

2. Return W, P, Q
```

## Prediction

To predict Y from new X:

```python
# Transform X to latent space
T = X @ W

# Predict Y
Y_pred = T @ Q.T
```

## Comparison: dPCA vs dPLS

| Aspect | dPCA | dPLS |
|--------|------|------|
| **Objective** | Max explained variance | Max covariance with Y |
| **Supervision** | Unsupervised | Supervised |
| **Output** | Variance decomposition | Predictions |
| **Best for** | Understanding structure | Predicting outcomes |

### Visual Comparison

```
dPCA: Find directions of max variance
      ┌───────────────────────────────┐
      │    ○ ○    ○                  │
      │  ○ ○ ○ ○ ○                   │  PC1 ─────→ Max variance
      │    ○ ○                       │
      └───────────────────────────────┘

dPLS: Find directions that predict Y
      ┌───────────────────────────────┐
      │ Low Y      High Y             │
      │    ●──────────────────○       │  PLS1 ─────→ Max cov(X,Y)
      │                               │
      └───────────────────────────────┘
```

## Multivariate Mental Scores

When Y has multiple columns (e.g., wellbeing, anxiety, stress):

```python
Y = np.array([
    [0.8, 0.2, 0.3],  # Subject 1: high wellbeing, low anxiety/stress
    [0.3, 0.7, 0.6],  # Subject 2: low wellbeing, high anxiety/stress
    ...
])
```

dPLS finds components that jointly predict all Y variables.

## Feature Importance

After fitting, extract which gait features contribute most:

```python
importance = dpls.get_feature_importance(marginalization='condition')
# {'hip_flexion': 0.35, 'stride_length': 0.28, ...}
```

This shows which gait features are most predictive of mental state.

## Regularization

dPLS includes regularization to prevent overfitting:

$$\mathcal{L} = ||Y - XW Q^T||^2 + λ||W||^2$$

Options:
- `regularizer='auto'`: Cross-validation to find optimal λ
- `regularizer=0.1`: Fixed value

## Code Example

```python
from src.dpca import DemixedPLS

# Data
X = np.random.randn(15, 100, 5)  # [features, time, conditions]
Y = np.random.randn(5, 3)        # [conditions, mental_vars]

# Fit
dpls = DemixedPLS(n_components=5, regularizer='auto')
dpls.fit(X, Y, 
         feature_labels=['hip', 'knee', ...],
         condition_labels=['neutral', 'anxious', ...])

# Predict
Y_pred = dpls.predict(X_new, marginalization='condition')

# Feature importance
importance = dpls.get_feature_importance('condition')
print(importance)  # {'hip': 0.35, 'knee': 0.22, ...}

# Explained variance
dpls.explained_variance_  # For each marginalization
```

## Use Cases

1. **Predicting mental state from gait**: Which gait features predict wellbeing?
2. **Clinical markers**: Find gait patterns associated with depression scores
3. **Fatigue detection**: Predict fatigue level from walking patterns

## When to Use dPLS vs dPCA

| Scenario | Use |
|----------|-----|
| Explore data structure | dPCA |
| Predict outcomes | dPLS |
| Understand variance sources | dPCA |
| Build predictive model | dPLS |
| No target variable | dPCA |
| Have target scores | dPLS |

## References

- Wold, H. (1985). Partial least squares. Encyclopedia of statistical sciences.
- Abdi, H. (2010). Partial least squares regression and projection on latent structure regression. Wiley interdisciplinary reviews.
