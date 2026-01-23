# ContinuousScoreDPCA

## Overview

**ContinuousScoreDPCA** is designed for analyzing the relationship between gait patterns and a **single continuous mental score** (e.g., wellbeing score from 0-100).

Unlike standard dPCA which requires categorical conditions, this method handles continuous, time-invariant mental scores.

## Problem Setting

- **Gait data**: `[n_subjects, n_features, n_timepoints]`
- **Mental score**: `[n_subjects]` - one value per subject (not varying over time)

Goal: Find gait patterns associated with the mental score.

## Key Insight

Mental scores like wellbeing don't change within a gait cycle - they're properties of the subject, not the moment.

```
Subject 1: Wellbeing = 0.8
  Gait cycle: ~~~~~~~~~~~~~~~~~~~~  (same score throughout)

Subject 2: Wellbeing = 0.3
  Gait cycle: ~~~~~~~~~~~~~~~~~~~~  (same score throughout)
```

## Mathematical Formulation

### 1. Score-Weighted Covariance

Instead of condition averages, we compute score-weighted statistics:

**Score-weighted mean**:
$$\bar{X}_s[n, t] = \sum_i s_i \cdot X_i[n, t] / \sum_i s_i$$

Where $s_i$ is the mental score for subject $i$.

**Score covariance**:
$$C_s = \sum_i (s_i - \bar{s})(X_i - \bar{X})^T (X_i - \bar{X})$$

### 2. Find Score-Associated Components

Find directions that maximize:
$$\max_w \text{Corr}(X \cdot w, s)$$

This finds gait patterns most correlated with the mental score.

### 3. Time-Averaged Features

Since the score is time-invariant, we often use time-averaged gait features:

$$\tilde{X}[i, n] = \frac{1}{T} \sum_t X_i[n, t]$$

Then correlate $\tilde{X}$ with scores.

## Algorithm

```
Input: 
  X[n_subjects, n_features, n_timepoints]
  scores[n_subjects]
  n_components

Output:
  score_components: Gait patterns associated with score
  score_weights: Feature weights for prediction

1. Center data and scores:
   X ← X - mean(X, axis=subjects)
   s ← scores - mean(scores)

2. Compute time-averaged features:
   X_avg[n_subjects, n_features] = mean(X, axis=time)

3. Compute score-feature correlations:
   For each feature n:
     correlations[n] = corr(X_avg[:, n], s)

4. Find principal score direction:
   a. Compute score covariance: C_s = X_avg.T @ diag(s) @ X_avg
   b. Compute feature covariance: C_X = X_avg.T @ X_avg
   c. Solve: C_s w = C_X w λ
   d. Select top k components

5. For prediction:
   score_weights = correlations / ||correlations||
   
6. Return score_components, score_weights, correlations
```

## Prediction

Predict mental score from new gait data:

```python
# Time-average the gait features
X_new_avg = X_new.mean(axis=2)  # [n_new_subjects, n_features]

# Predict using learned weights
predicted_score = X_new_avg @ score_weights
```

## Feature Importance

Identify which gait features are most associated with wellbeing:

```python
model = ContinuousScoreDPCA(n_components=5)
model.fit(gait_data, wellbeing_scores)

# Get feature correlations with score
summary = model.summary()
# {'hip_flexion': 0.45, 'stride_length': 0.62, ...}
```

Interpretation:
- **Positive correlation**: Higher feature value → Higher wellbeing
- **Negative correlation**: Higher feature value → Lower wellbeing

## Temporal Analysis

Although the score is time-invariant, we can analyze how the association varies across the gait cycle:

```python
# Correlation at each timepoint
temporal_corr = np.zeros((n_features, n_timepoints))
for t in range(n_timepoints):
    for n in range(n_features):
        temporal_corr[n, t] = corr(X[:, n, t], scores)
```

This shows when during the gait cycle each feature is most predictive.

## Code Example

```python
from src.dpca import ContinuousScoreDPCA
import numpy as np

# Data
gait_data = np.random.randn(50, 15, 100)  # [subjects, features, time]
wellbeing_scores = np.random.rand(50)      # [subjects]

# Fit
model = ContinuousScoreDPCA(n_components=5)
model.fit(
    gait_data, 
    wellbeing_scores,
    feature_labels=['hip_flexion', 'knee_flexion', 'stride_length', ...]
)

# Get feature importance
summary = model.summary()
print("Most predictive features:")
for feat, corr in sorted(summary.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
    print(f"  {feat}: r = {corr:.3f}")

# Predict on new subjects
new_gait = np.random.randn(10, 15, 100)
predicted_wellbeing = model.predict_score(new_gait)
```

## Visualization

```python
import matplotlib.pyplot as plt

# Feature-score correlations
plt.figure(figsize=(10, 6))
correlations = model.feature_correlations_
plt.barh(model.feature_labels_, correlations)
plt.xlabel('Correlation with Wellbeing')
plt.ylabel('Gait Feature')
plt.axvline(0, color='k', linestyle='--')
plt.title('Gait Features Associated with Wellbeing')
plt.tight_layout()
plt.show()
```

## Comparison with Regression

| Aspect | ContinuousScoreDPCA | Linear Regression |
|--------|---------------------|-------------------|
| **Goal** | Find patterns | Predict score |
| **Output** | Components + weights | Coefficients |
| **Interpretation** | Latent structure | Direct effects |
| **Multicollinearity** | Handled via PCA | Can be problematic |

## Use Cases

1. **Wellbeing prediction**: Which gait patterns indicate higher wellbeing?
2. **Depression screening**: Gait markers of depression severity
3. **Fatigue monitoring**: Predict fatigue level from walking
4. **Pain assessment**: Gait changes associated with pain scores

## Limitations

1. **Linear relationships only**: Assumes linear association between gait and score
2. **Cross-sectional**: Doesn't capture within-subject changes over time
3. **Single score**: For multiple mental variables, use MultiVariateMentalDPCA

## When to Use

| Data Structure | Method |
|----------------|--------|
| Single continuous score | ContinuousScoreDPCA |
| Multiple continuous scores | MultiVariateMentalDPCA |
| Categorical conditions | DemixedPCA |
| Predict scores | DemixedPLS |
