"""
Integration tests for gait_latent_dynamics

These tests verify end-to-end workflows combining multiple modules:
- Data generation -> Model fitting -> Analysis -> Visualization
- File I/O roundtrips
- Cross-module compatibility

Integration tests are marked with @pytest.mark.integration
Run with: pytest -m integration
"""

import sys
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dpca import (
    DemixedPCA, DemixedPLS, GaitDPCA,
    ContinuousScoreDPCA, MultiVariateMentalDPCA
)
from src.data_loader import (
    generate_synthetic_gait_data, GaitDataLoader,
    create_gait_cycle_time_axis
)
from src.visualization import DPCAVisualizer, quick_plot_dpca_results


@pytest.fixture(autouse=True)
def close_figures():
    """Automatically close all figures after each test"""
    yield
    plt.close('all')


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================

@pytest.mark.integration
class TestFullAnalysisPipeline:
    """Test complete analysis pipelines from data generation to visualization"""

    def test_dpca_full_pipeline(self):
        """Test: generate data -> fit dPCA -> analyze -> visualize"""
        # Step 1: Generate synthetic data
        data, metadata = generate_synthetic_gait_data(
            n_trials=30,
            n_features=10,
            n_timepoints=50,
            n_conditions=4,
            random_state=42
        )

        assert data.shape == (30, 10, 50, 4)

        # Step 2: Compute trial-averaged data
        trial_averaged = data.mean(axis=0)
        assert trial_averaged.shape == (10, 50, 4)

        # Step 3: Fit DemixedPCA
        dpca = DemixedPCA(n_components=5)
        dpca.fit(trial_averaged)

        assert 'time' in dpca.components_
        assert 'condition' in dpca.components_
        assert 'interaction' in dpca.components_

        # Step 4: Transform and analyze
        transformed = dpca.transform(trial_averaged)
        # transform() returns a dict when no marginalization specified
        assert isinstance(transformed, dict)
        assert 'time' in transformed

        # Step 5: Get demixing summary
        summary = dpca.get_demixing_summary()
        assert sum(summary.values()) > 0

        # Step 6: Inverse transform (reconstruction)
        reconstructed = dpca.inverse_transform(
            dpca.transform(trial_averaged, marginalization='time'),
            marginalization='time'
        )
        assert reconstructed.shape == trial_averaged.shape

        # Step 7: Visualize
        viz = DPCAVisualizer()
        fig = viz.plot_explained_variance(dpca)
        assert isinstance(fig, plt.Figure)

        # Step 8: Comprehensive visualization
        fig2 = viz.plot_comprehensive_summary(
            dpca, trial_averaged, metadata['mental_state_labels']
        )
        assert isinstance(fig2, plt.Figure)

    def test_dpls_prediction_pipeline(self):
        """Test: generate data -> fit dPLS -> predict -> evaluate"""
        # Step 1: Generate data with mental state effect
        data, metadata = generate_synthetic_gait_data(
            n_trials=50,
            n_features=8,
            n_timepoints=40,
            n_conditions=4,
            mental_state_effect_strength=0.5,
            noise_level=0.1,
            random_state=42
        )

        # Step 2: Create target values (mental state scores)
        Y = np.array([1.0, 2.0, 3.0, 4.0])  # Ordered mental states

        # Step 3: Trial-average and fit
        X = data.mean(axis=0)
        dpls = DemixedPLS(n_components=3)
        dpls.fit(X, Y, feature_labels=metadata['feature_labels'])

        # Step 4: Predict
        Y_pred = dpls.predict(X)
        assert Y_pred.shape == Y.shape

        # Step 5: Check correlation (predictions should be somewhat correlated)
        corr = np.corrcoef(Y, Y_pred)[0, 1]
        # With strong effect and low noise, should have decent correlation
        assert corr > 0.5

        # Step 6: Get feature importance
        importance = dpls.get_feature_importance('condition')
        assert len(importance) == 8

        # Step 7: Get summary
        summary = dpls.summary()
        assert 'top_features' in summary

    def test_continuous_score_pipeline(self):
        """Test: generate subject data -> fit ContinuousScoreDPCA -> predict scores"""
        np.random.seed(42)

        # Step 1: Create subject-level data
        n_subjects = 30
        n_features = 8
        n_timepoints = 50

        # Create mental scores
        scores = np.random.uniform(3, 8, n_subjects)

        # Create gait data correlated with scores
        t = np.linspace(0, 2*np.pi, n_timepoints)
        X = np.zeros((n_subjects, n_features, n_timepoints))

        for i in range(n_subjects):
            for f in range(n_features):
                base_pattern = np.sin(t + f * 0.3)
                score_effect = 0.3 * (scores[i] - 5) * (1 if f < 4 else 0)
                X[i, f, :] = base_pattern + score_effect + np.random.randn(n_timepoints) * 0.1

        # Step 2: Fit model
        model = ContinuousScoreDPCA(n_components=5)
        model.fit(X, scores, feature_labels=[f'f{i}' for i in range(n_features)])

        # Step 3: Predict scores
        predicted = model.predict_score(X)
        assert predicted.shape == scores.shape

        # Step 4: Check prediction quality
        corr = np.corrcoef(scores, predicted)[0, 1]
        assert corr > 0.3  # Should have some predictive power

        # Step 5: Get score-related features
        related = model.get_score_related_features(threshold=0.2)
        assert isinstance(related, list)

        # Step 6: Get summary
        summary = model.summary()
        assert 'top_score_related_features' in summary

    def test_multivariate_mental_pipeline(self):
        """Test: multivariate gait-mental analysis pipeline"""
        np.random.seed(42)

        # Step 1: Create data
        n_subjects = 35
        n_timepoints = 50
        n_body_factors = 10
        n_mental_vars = 4

        # Mental scores
        mental = np.random.randn(n_subjects, n_mental_vars)

        # Gait data with mental correlations
        t = np.linspace(0, 2*np.pi, n_timepoints)
        gait = np.zeros((n_subjects, n_timepoints, n_body_factors))

        for i in range(n_subjects):
            for f in range(n_body_factors):
                base = np.sin(t + f * 0.3)
                # Add correlation with mental variables
                mental_effect = 0.2 * mental[i, f % n_mental_vars]
                gait[i, :, f] = base + mental_effect + np.random.randn(n_timepoints) * 0.1

        # Step 2: Fit model
        model = MultiVariateMentalDPCA(n_gait_components=5, method='cca')
        model.fit(
            gait, mental,
            gait_labels=[f'body_{i}' for i in range(n_body_factors)],
            mental_labels=['wellbeing', 'anxiety', 'stress', 'fatigue']
        )

        # Step 3: Check canonical correlations
        assert model.correlations_ is not None
        assert len(model.correlations_) > 0

        # Step 4: Predict mental from gait
        predicted_mental = model.predict_mental(gait)
        assert predicted_mental.shape == mental.shape

        # Step 5: Get associations
        associations = model.get_mental_gait_associations()
        assert isinstance(associations, dict)

        # Step 6: Get interpretations
        interpretations = model.get_canonical_interpretation()
        assert len(interpretations) > 0

    def test_gait_dpca_pipeline(self):
        """Test: GaitDPCA specialized analysis pipeline"""
        # Step 1: Generate data
        data, metadata = generate_synthetic_gait_data(
            n_trials=40,
            n_features=10,
            n_timepoints=50,
            n_conditions=3,
            random_state=42
        )

        X = data.mean(axis=0)

        # Step 2: Fit GaitDPCA
        model = GaitDPCA(n_components=5)
        model.fit_with_labels(
            X,
            mental_state_labels=metadata['mental_state_labels'],
            feature_labels=metadata['feature_labels']
        )

        # Step 3: Extract components
        mental_comps = model.get_mental_state_components()
        time_comps = model.get_gait_phase_components()
        interaction_comps = model.get_interaction_components()

        assert mental_comps is not None
        assert time_comps is not None
        assert interaction_comps is not None

        # Step 4: Analyze mental state separation
        separation = model.analyze_mental_state_separation(X, metadata['mental_state_labels'])
        # Returns a dict with pairwise distances between mental states
        assert isinstance(separation, dict)
        assert len(separation) > 0

        # Step 5: Visualize
        viz = DPCAVisualizer()
        # GaitDPCA uses 'condition' marginalization (not 'mental_state')
        fig = viz.plot_mental_state_separation(
            model, X, metadata['mental_state_labels'],
            marginalization='condition'
        )
        assert isinstance(fig, plt.Figure)


# =============================================================================
# Data I/O Roundtrip Tests
# =============================================================================

@pytest.mark.integration
class TestDataIOIntegration:
    """Test data saving and loading roundtrips"""

    def test_numpy_save_load_roundtrip(self):
        """Test saving and loading numpy data preserves integrity"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate data
            data, metadata = generate_synthetic_gait_data(
                n_trials=20, n_features=8, n_timepoints=50, n_conditions=3,
                random_state=42
            )

            # Save
            loader = GaitDataLoader()
            filepath = Path(tmpdir) / 'test_data.npz'
            loader.save(data, str(filepath), metadata=metadata)

            # Load
            loaded_data, loaded_meta = loader.load_from_numpy(str(filepath))

            # Verify
            np.testing.assert_array_equal(loaded_data, data)

    def test_preprocess_then_analyze(self):
        """Test preprocessing followed by analysis"""
        # Generate raw data
        data, metadata = generate_synthetic_gait_data(
            n_trials=30, n_features=10, n_timepoints=100, n_conditions=4,
            noise_level=0.3,
            random_state=42
        )

        # Preprocess
        loader = GaitDataLoader()
        processed = loader.preprocess(
            data,
            normalize=True,
            filter_cutoff=0.2,
            resample_points=50
        )

        assert processed.shape == (30, 10, 50, 4)
        assert np.abs(processed.mean()) < 0.1  # Normalized

        # Analyze
        X = processed.mean(axis=0)
        dpca = DemixedPCA(n_components=3)
        dpca.fit(X)

        # Should still work properly
        assert 'time' in dpca.components_


# =============================================================================
# Cross-Module Compatibility Tests
# =============================================================================

@pytest.mark.integration
class TestCrossModuleCompatibility:
    """Test that different modules work together correctly"""

    def test_data_loader_to_dpca_to_visualization(self):
        """Test full chain: data loading -> dPCA -> visualization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Generate and save data
            data, metadata = generate_synthetic_gait_data(
                n_trials=25, n_features=8, n_timepoints=50, n_conditions=4,
                random_state=42
            )

            loader = GaitDataLoader()
            filepath = Path(tmpdir) / 'gait_data.npz'
            loader.save(data, str(filepath), metadata=metadata)

            # Step 2: Load data
            loaded_data, loaded_meta = loader.load_from_numpy(str(filepath))

            # Step 3: Preprocess
            processed = loader.preprocess(loaded_data, normalize=True)

            # Step 4: Fit dPCA
            X = loader.get_trial_averaged_data(processed)
            dpca = DemixedPCA(n_components=3)
            dpca.fit(X)

            # Step 5: Visualize and save
            output_dir = Path(tmpdir) / 'figures'
            figures = quick_plot_dpca_results(
                dpca, X, loaded_meta.get('mental_state_labels', ['a', 'b', 'c', 'd']),
                feature_labels=loaded_meta.get('feature_labels'),
                save_dir=str(output_dir)
            )

            # Verify outputs
            assert len(figures) >= 3
            assert output_dir.exists()
            assert len(list(output_dir.glob('*.png'))) >= 3

    def test_synthetic_data_consistency(self):
        """Test that synthetic data works with all model types"""
        # Generate consistent data
        data, metadata = generate_synthetic_gait_data(
            n_trials=40, n_features=10, n_timepoints=50, n_conditions=4,
            random_state=42
        )

        X_3d = data.mean(axis=0)

        # Test DemixedPCA
        dpca = DemixedPCA(n_components=3)
        dpca.fit(X_3d)
        assert dpca.components_ is not None

        # Test DemixedPLS
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        dpls = DemixedPLS(n_components=3)
        dpls.fit(X_3d, Y)
        assert dpls.predict(X_3d) is not None

        # Test GaitDPCA
        gait_dpca = GaitDPCA(n_components=3)
        gait_dpca.fit_with_labels(
            X_3d,
            mental_state_labels=metadata['mental_state_labels'],
            feature_labels=metadata['feature_labels']
        )
        assert gait_dpca.get_mental_state_components() is not None


# =============================================================================
# Performance and Stability Tests
# =============================================================================

@pytest.mark.integration
class TestPerformanceAndStability:
    """Tests for performance and numerical stability across workflows"""

    def test_repeated_fit_consistency(self):
        """Test that repeated fitting produces consistent results"""
        np.random.seed(42)
        data = np.random.randn(10, 50, 4)

        results = []
        for _ in range(3):
            dpca = DemixedPCA(n_components=3)
            dpca.fit(data)
            results.append(dpca.explained_variance_ratio_['time'][0])

        # Results should be identical (deterministic)
        assert all(r == results[0] for r in results)

    def test_significance_analysis_integration(self):
        """Test significance analysis in full pipeline"""
        # Generate data with clear effect
        data, metadata = generate_synthetic_gait_data(
            n_trials=30,
            n_features=8,
            n_timepoints=40,
            n_conditions=4,
            mental_state_effect_strength=0.8,
            noise_level=0.1,
            random_state=42
        )

        # Fit on trial-averaged
        X = data.mean(axis=0)
        dpca = DemixedPCA(n_components=3)
        dpca.fit(X)

        # Run significance analysis on trial-level data
        significance = dpca.significance_analysis(data, n_shuffles=50, alpha=0.05)

        assert 'condition' in significance
        assert 'time' in significance
        assert 'interaction' in significance

        # With strong effect, at least some components should be significant
        # (though this is stochastic, so we just check structure)
        for key, sig_array in significance.items():
            assert len(sig_array) == 3  # n_components

    def test_large_dataset_handling(self):
        """Test handling of larger datasets"""
        # Generate larger dataset
        data, metadata = generate_synthetic_gait_data(
            n_trials=100,
            n_features=15,
            n_timepoints=100,
            n_conditions=5,
            random_state=42
        )

        # Should handle without memory issues
        loader = GaitDataLoader()
        processed = loader.preprocess(data, normalize=True, resample_points=50)
        X = loader.get_trial_averaged_data(processed)

        dpca = DemixedPCA(n_components=5)
        dpca.fit(X)

        assert dpca.components_['time'].shape[0] == 15  # n_features

    def test_reconstruction_quality(self):
        """Test that reconstruction maintains reasonable fidelity"""
        np.random.seed(42)
        data = np.random.randn(10, 50, 4)

        dpca = DemixedPCA(n_components=5)
        dpca.fit(data)

        # Full reconstruction should be close to original
        all_transformed = {}
        for margin in ['time', 'condition', 'interaction']:
            all_transformed[margin] = dpca.transform(data, marginalization=margin)

        reconstructed = dpca.inverse_transform(all_transformed)

        # Check reconstruction error
        mse = np.mean((data - reconstructed) ** 2)
        original_var = np.var(data)

        # Reconstruction should capture a reasonable amount of variance
        # With limited components, some variance loss is expected
        assert mse < original_var * 0.8  # Less than 80% of variance lost


def run_quick_tests():
    """Run quick integration tests without pytest"""
    print("=" * 60)
    print("Running Quick Integration Tests")
    print("=" * 60)

    matplotlib.use('Agg')
    np.random.seed(42)

    # Test 1: Full dPCA pipeline
    print("\n[1] Testing full dPCA pipeline...")
    try:
        data, meta = generate_synthetic_gait_data(
            n_trials=20, n_features=8, n_timepoints=50, n_conditions=4,
            random_state=42
        )
        X = data.mean(axis=0)
        dpca = DemixedPCA(n_components=3)
        dpca.fit(X)
        transformed = dpca.transform(X)
        summary = dpca.get_demixing_summary()

        viz = DPCAVisualizer()
        fig = viz.plot_explained_variance(dpca)
        plt.close(fig)

        print(f"    ✓ Pipeline complete. Variance summary: {summary}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Test 2: Data I/O roundtrip
    print("\n[2] Testing data I/O roundtrip...")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            data, meta = generate_synthetic_gait_data(n_trials=10, random_state=42)
            loader = GaitDataLoader()
            filepath = Path(tmpdir) / 'test.npz'
            loader.save(data, str(filepath), metadata=meta)
            loaded, _ = loader.load_from_numpy(str(filepath))
            assert np.allclose(data, loaded)
            print("    ✓ Data I/O roundtrip successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Test 3: Cross-module compatibility
    print("\n[3] Testing cross-module compatibility...")
    try:
        data, meta = generate_synthetic_gait_data(n_trials=20, random_state=42)
        loader = GaitDataLoader()
        processed = loader.preprocess(data, normalize=True)
        X = loader.get_trial_averaged_data(processed)

        dpca = DemixedPCA(n_components=3)
        dpca.fit(X)

        viz = DPCAVisualizer()
        fig = viz.plot_comprehensive_summary(dpca, X, meta['mental_state_labels'])
        plt.close(fig)
        print("    ✓ Cross-module workflow successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Test 4: Multiple model types
    print("\n[4] Testing multiple model types...")
    try:
        data, meta = generate_synthetic_gait_data(n_trials=30, random_state=42)
        X = data.mean(axis=0)

        # DemixedPCA
        dpca = DemixedPCA(n_components=3)
        dpca.fit(X)

        # DemixedPLS
        Y = np.array([1, 2, 3, 4])
        dpls = DemixedPLS(n_components=3)
        dpls.fit(X, Y)

        # GaitDPCA
        gait_dpca = GaitDPCA(n_components=3)
        gait_dpca.fit_with_labels(X, meta['mental_state_labels'], meta['feature_labels'])

        print("    ✓ All model types working")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("Quick Integration Tests Complete!")
    print("=" * 60)


if __name__ == '__main__':
    run_quick_tests()
