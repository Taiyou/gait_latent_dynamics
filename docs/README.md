# Documentation

Detailed explanations of each method in the Gait Latent Dynamics package.

## 📘 Practical Guide

**[Gait Latent Dynamics Guide](gait_latent_dynamics_guide.md)** - Complete tutorial for analyzing gait-mental relationships:
- Gait data overview and features
- Mental state variables
- Step-by-step workflow
- Method selection guide
- Practical code examples
- Interpretation guidelines

---

## Methods Overview

| Method | Use Case | Mental Variable Type |
|--------|----------|---------------------|
| [DemixedPCA](demixed_pca.md) | Separate variance by source | Categorical conditions |
| [DemixedPLS](demixed_pls.md) | Predict mental scores | Continuous (supervised) |
| [ContinuousScoreDPCA](continuous_score_dpca.md) | Single mental score analysis | Single continuous |
| [MultiVariateMentalDPCA](multivariate_mental_dpca.md) | Multi-variable analysis | Multiple continuous |

## Quick Selection Guide

```
Do you have mental state labels?
├── No → Use standard PCA
└── Yes → What type?
    ├── Categorical (e.g., "anxious", "relaxed")
    │   ├── Want to understand structure? → DemixedPCA
    │   └── Want to predict? → DemixedPLS
    └── Continuous scores (e.g., 0-100)
        ├── Single variable (e.g., wellbeing only) → ContinuousScoreDPCA
        └── Multiple variables (wellbeing, anxiety, ...) → MultiVariateMentalDPCA
```

## Data Structure Comparison

### Subject-Level (Recommended)

```
Gait: [n_subjects, n_timepoints, n_body_factors]
Mental: [n_subjects, n_mental_vars]

Methods: MultiVariateMentalDPCA, ContinuousScoreDPCA
```

### Condition-Based (Traditional dPCA)

```
Gait: [n_features, n_timepoints, n_conditions]
Mental: Categorical labels for conditions

Methods: DemixedPCA, DemixedPLS
```

## Mathematical Foundations

| Method | Core Technique | Objective |
|--------|---------------|-----------|
| DemixedPCA | Eigenvalue decomposition | Maximize variance per source |
| DemixedPLS | NIPALS algorithm | Maximize covariance with Y |
| ContinuousScoreDPCA | Score-weighted PCA | Maximize correlation with score |
| MultiVariateMentalDPCA | CCA / PLS | Find multivariate associations |

## Detailed Documentation

1. **[Demixed PCA](demixed_pca.md)**
   - Marginalization concept
   - Variance decomposition
   - Regularization
   - Comparison with standard PCA

2. **[Demixed PLS](demixed_pls.md)**
   - NIPALS algorithm
   - Prediction workflow
   - Feature importance
   - Comparison with dPCA

3. **[ContinuousScoreDPCA](continuous_score_dpca.md)**
   - Score-weighted covariance
   - Feature-score correlations
   - Prediction model
   - Single variable focus

4. **[MultiVariateMentalDPCA](multivariate_mental_dpca.md)**
   - CCA formulation
   - Gait-mental associations
   - Correlation heatmaps
   - Multiple variable handling
