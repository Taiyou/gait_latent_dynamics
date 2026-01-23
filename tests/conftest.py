"""
Pytest configuration and shared fixtures for gait_latent_dynamics tests

This file provides common fixtures used across multiple test modules.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.dpca import DemixedPCA, DemixedPLS, GaitDPCA, ContinuousScoreDPCA, MultiVariateMentalDPCA
from src.data_loader import generate_synthetic_gait_data


# =============================================================================
# Random State Fixtures
# =============================================================================

@pytest.fixture
def random_seed():
    """Set random seed for reproducibility"""
    np.random.seed(42)
    return 42


# =============================================================================
# Basic Data Fixtures
# =============================================================================

@pytest.fixture
def sample_3d_data():
    """
    Standard 3D test data: (n_features, n_timepoints, n_conditions)
    Shape: (10, 50, 4)
    """
    np.random.seed(42)
    n_features = 10
    n_timepoints = 50
    n_conditions = 4

    t = np.linspace(0, 2*np.pi, n_timepoints)
    data = np.zeros((n_features, n_timepoints, n_conditions))

    for f in range(n_features):
        for c in range(n_conditions):
            data[f, :, c] = np.sin(t + f*0.3) + 0.3*(c-1.5) + np.random.randn(n_timepoints)*0.1

    return data


@pytest.fixture
def sample_4d_data():
    """
    4D trial-level test data: (n_trials, n_features, n_timepoints, n_conditions)
    Shape: (20, 10, 50, 4)
    """
    np.random.seed(42)
    n_trials = 20
    n_features = 10
    n_timepoints = 50
    n_conditions = 4

    t = np.linspace(0, 2*np.pi, n_timepoints)
    data = np.zeros((n_trials, n_features, n_timepoints, n_conditions))

    for trial in range(n_trials):
        for f in range(n_features):
            for c in range(n_conditions):
                data[trial, f, :, c] = (
                    np.sin(t + f*0.3) +
                    0.3*(c-1.5) +
                    np.random.randn(n_timepoints)*0.1
                )

    return data


@pytest.fixture
def sample_subject_data():
    """
    Subject-level gait data: (n_subjects, n_features, n_timepoints)
    Shape: (25, 8, 50)
    """
    np.random.seed(42)
    n_subjects = 25
    n_features = 8
    n_timepoints = 50

    t = np.linspace(0, 2*np.pi, n_timepoints)
    data = np.zeros((n_subjects, n_features, n_timepoints))

    for i in range(n_subjects):
        for f in range(n_features):
            data[i, f, :] = np.sin(t + f*0.3) + np.random.randn(n_timepoints)*0.1

    return data


@pytest.fixture
def sample_multivariate_gait_data():
    """
    Gait data for MultiVariateMentalDPCA: (n_subjects, n_timepoints, n_body_factors)
    Shape: (30, 50, 10)
    """
    np.random.seed(42)
    n_subjects = 30
    n_timepoints = 50
    n_body_factors = 10

    t = np.linspace(0, 2*np.pi, n_timepoints)
    data = np.zeros((n_subjects, n_timepoints, n_body_factors))

    for i in range(n_subjects):
        for f in range(n_body_factors):
            data[i, :, f] = np.sin(t + f*0.3) + np.random.randn(n_timepoints)*0.1

    return data


# =============================================================================
# Target/Score Fixtures
# =============================================================================

@pytest.fixture
def sample_condition_scores():
    """
    Mental scores per condition
    Shape: (4,) - matches n_conditions in sample_3d_data
    """
    return np.array([3.0, 5.0, 7.0, 4.0])


@pytest.fixture
def sample_subject_scores():
    """
    Mental scores per subject
    Shape: (25,) - matches n_subjects in sample_subject_data
    """
    np.random.seed(42)
    return np.random.uniform(3, 8, 25)


@pytest.fixture
def sample_mental_scores():
    """
    Multi-variate mental scores: (n_subjects, n_mental_vars)
    Shape: (30, 4) - matches n_subjects in sample_multivariate_gait_data
    """
    np.random.seed(42)
    return np.random.randn(30, 4)


# =============================================================================
# Label Fixtures
# =============================================================================

@pytest.fixture
def feature_labels_10():
    """Feature labels for 10 features"""
    return [f'feature_{i}' for i in range(10)]


@pytest.fixture
def feature_labels_8():
    """Feature labels for 8 features"""
    return [f'feature_{i}' for i in range(8)]


@pytest.fixture
def mental_state_labels_4():
    """Mental state labels for 4 conditions"""
    return ['neutral', 'anxious', 'relaxed', 'focused']


@pytest.fixture
def mental_state_labels_3():
    """Mental state labels for 3 conditions"""
    return ['low', 'medium', 'high']


@pytest.fixture
def mental_variable_labels():
    """Labels for mental variables in multivariate analysis"""
    return ['wellbeing', 'anxiety', 'stress', 'fatigue']


@pytest.fixture
def body_factor_labels():
    """Labels for body factors"""
    return [f'body_factor_{i}' for i in range(10)]


# =============================================================================
# Fitted Model Fixtures
# =============================================================================

@pytest.fixture
def fitted_dpca(sample_3d_data):
    """Pre-fitted DemixedPCA model"""
    dpca = DemixedPCA(n_components=3)
    dpca.fit(sample_3d_data)
    return dpca


@pytest.fixture
def fitted_dpls(sample_3d_data, sample_condition_scores, feature_labels_10):
    """Pre-fitted DemixedPLS model"""
    dpls = DemixedPLS(n_components=3)
    dpls.fit(sample_3d_data, sample_condition_scores, feature_labels=feature_labels_10)
    return dpls


@pytest.fixture
def fitted_gait_dpca(sample_3d_data, mental_state_labels_4, feature_labels_10):
    """Pre-fitted GaitDPCA model"""
    # Use data with 4 conditions
    np.random.seed(42)
    data = np.random.randn(10, 50, 4)

    model = GaitDPCA(n_components=3)
    model.fit_with_labels(
        data,
        mental_state_labels=mental_state_labels_4,
        feature_labels=feature_labels_10
    )
    return model


@pytest.fixture
def fitted_continuous_score_dpca(sample_subject_data, sample_subject_scores, feature_labels_8):
    """Pre-fitted ContinuousScoreDPCA model"""
    model = ContinuousScoreDPCA(n_components=3)
    model.fit(sample_subject_data, sample_subject_scores, feature_labels=feature_labels_8)
    return model


@pytest.fixture
def fitted_multivariate_dpca(sample_multivariate_gait_data, sample_mental_scores,
                              mental_variable_labels, body_factor_labels):
    """Pre-fitted MultiVariateMentalDPCA model"""
    model = MultiVariateMentalDPCA(n_gait_components=5, method='cca')
    model.fit(
        sample_multivariate_gait_data,
        sample_mental_scores,
        gait_labels=body_factor_labels,
        mental_labels=mental_variable_labels
    )
    return model


# =============================================================================
# Synthetic Data Fixtures
# =============================================================================

@pytest.fixture
def synthetic_gait_data():
    """
    Generate synthetic gait data using the data_loader function
    Returns: (data, metadata)
    """
    data, metadata = generate_synthetic_gait_data(
        n_trials=30,
        n_features=10,
        n_timepoints=50,
        n_conditions=4,
        mental_state_effect_strength=0.3,
        time_effect_strength=0.5,
        interaction_strength=0.15,
        noise_level=0.1,
        random_state=42
    )
    return data, metadata


@pytest.fixture
def synthetic_gait_data_clean():
    """
    Synthetic gait data with minimal noise (for testing effect detection)
    """
    data, metadata = generate_synthetic_gait_data(
        n_trials=50,
        n_features=10,
        n_timepoints=50,
        n_conditions=4,
        mental_state_effect_strength=0.5,
        time_effect_strength=0.5,
        interaction_strength=0.2,
        noise_level=0.01,
        random_state=42
    )
    return data, metadata


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def time_axis_50():
    """Time axis for 50 timepoints (0-100% gait cycle)"""
    return np.linspace(0, 100, 50)


@pytest.fixture
def time_axis_100():
    """Time axis for 100 timepoints (0-100% gait cycle)"""
    return np.linspace(0, 100, 100)


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Pytest configuration hook"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# =============================================================================
# Session-scoped Fixtures (for expensive operations)
# =============================================================================

@pytest.fixture(scope="session")
def large_synthetic_data():
    """
    Large synthetic dataset for performance testing (session-scoped to avoid regeneration)
    """
    data, metadata = generate_synthetic_gait_data(
        n_trials=100,
        n_features=15,
        n_timepoints=100,
        n_conditions=5,
        random_state=42
    )
    return data, metadata
