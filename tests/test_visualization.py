"""
Test suite for visualization module

Tests for:
- DPCAVisualizer class
- quick_plot_dpca_results function

Note: These tests verify that visualization functions:
1. Return correct types (matplotlib Figure objects)
2. Handle save paths correctly
3. Accept valid parameters without errors
4. Create expected subplot structures

Visual correctness is not automatically verified - these are smoke tests.
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
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from src.visualization import DPCAVisualizer, quick_plot_dpca_results
from src.dpca import DemixedPCA, GaitDPCA
from src.data_loader import generate_synthetic_gait_data


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample 3D data for visualization tests"""
    np.random.seed(42)
    return np.random.randn(10, 50, 4)


@pytest.fixture
def fitted_dpca(sample_data):
    """Create a fitted DemixedPCA model"""
    dpca = DemixedPCA(n_components=3)
    dpca.fit(sample_data)
    return dpca


@pytest.fixture
def mental_state_labels():
    """Sample mental state labels"""
    return ['neutral', 'anxious', 'relaxed', 'focused']


@pytest.fixture
def feature_labels():
    """Sample feature labels"""
    return [f'feature_{i}' for i in range(10)]


@pytest.fixture
def time_axis():
    """Sample time axis"""
    return np.linspace(0, 100, 50)


@pytest.fixture(autouse=True)
def close_figures():
    """Automatically close all figures after each test to prevent memory leaks"""
    yield
    plt.close('all')


# =============================================================================
# DPCAVisualizer Initialization Tests
# =============================================================================

class TestDPCAVisualizerInit:
    """Tests for DPCAVisualizer initialization"""

    def test_default_initialization(self):
        """Test default initialization parameters"""
        viz = DPCAVisualizer()
        assert viz.figsize == (12, 8)
        assert viz.dpi == 100

    def test_custom_figsize(self):
        """Test custom figsize"""
        viz = DPCAVisualizer(figsize=(16, 10))
        assert viz.figsize == (16, 10)

    def test_custom_dpi(self):
        """Test custom dpi"""
        viz = DPCAVisualizer(dpi=150)
        assert viz.dpi == 150

    def test_invalid_style_fallback(self):
        """Test that invalid style falls back gracefully"""
        # Should not raise error, falls back to default
        viz = DPCAVisualizer(style='nonexistent_style')
        assert viz is not None

    def test_color_palettes_defined(self):
        """Test that color palettes are defined"""
        assert hasattr(DPCAVisualizer, 'MENTAL_STATE_COLORS')
        assert hasattr(DPCAVisualizer, 'COMPONENT_COLORS')
        assert 'neutral' in DPCAVisualizer.MENTAL_STATE_COLORS
        assert 'time' in DPCAVisualizer.COMPONENT_COLORS


# =============================================================================
# plot_explained_variance Tests
# =============================================================================

class TestPlotExplainedVariance:
    """Tests for plot_explained_variance method"""

    def test_returns_figure(self, fitted_dpca):
        """Test that method returns a matplotlib Figure"""
        viz = DPCAVisualizer()
        fig = viz.plot_explained_variance(fitted_dpca)
        assert isinstance(fig, plt.Figure)

    def test_custom_figsize(self, fitted_dpca):
        """Test custom figsize parameter"""
        viz = DPCAVisualizer()
        fig = viz.plot_explained_variance(fitted_dpca, figsize=(8, 6))
        assert fig.get_size_inches()[0] == 8
        assert fig.get_size_inches()[1] == 6

    def test_has_two_subplots(self, fitted_dpca):
        """Test that figure has two subplots"""
        viz = DPCAVisualizer()
        fig = viz.plot_explained_variance(fitted_dpca)
        assert len(fig.axes) == 2

    def test_save_path(self, fitted_dpca):
        """Test saving figure to file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = DPCAVisualizer()
            save_path = Path(tmpdir) / 'explained_variance.png'
            fig = viz.plot_explained_variance(fitted_dpca, save_path=str(save_path))
            assert save_path.exists()
            assert save_path.stat().st_size > 0


# =============================================================================
# plot_component_timecourse Tests
# =============================================================================

class TestPlotComponentTimecourse:
    """Tests for plot_component_timecourse method"""

    def test_returns_figure(self, fitted_dpca, sample_data, mental_state_labels):
        """Test that method returns a matplotlib Figure"""
        viz = DPCAVisualizer()
        fig = viz.plot_component_timecourse(
            fitted_dpca, sample_data, 'condition',
            components=[0, 1],
            mental_state_labels=mental_state_labels
        )
        assert isinstance(fig, plt.Figure)

    def test_correct_number_of_subplots(self, fitted_dpca, sample_data, mental_state_labels):
        """Test that number of subplots matches number of components"""
        viz = DPCAVisualizer()

        # 2 components
        fig = viz.plot_component_timecourse(
            fitted_dpca, sample_data, 'condition',
            components=[0, 1],
            mental_state_labels=mental_state_labels
        )
        assert len(fig.axes) == 2
        plt.close(fig)

        # 3 components
        fig = viz.plot_component_timecourse(
            fitted_dpca, sample_data, 'time',
            components=[0, 1, 2],
            mental_state_labels=mental_state_labels
        )
        assert len(fig.axes) == 3

    def test_custom_time_axis(self, fitted_dpca, sample_data, mental_state_labels, time_axis):
        """Test with custom time axis"""
        viz = DPCAVisualizer()
        fig = viz.plot_component_timecourse(
            fitted_dpca, sample_data, 'condition',
            components=[0],
            mental_state_labels=mental_state_labels,
            time_axis=time_axis
        )
        assert isinstance(fig, plt.Figure)

    def test_save_path(self, fitted_dpca, sample_data, mental_state_labels):
        """Test saving figure to file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = DPCAVisualizer()
            save_path = Path(tmpdir) / 'timecourse.png'
            fig = viz.plot_component_timecourse(
                fitted_dpca, sample_data, 'condition',
                components=[0],
                mental_state_labels=mental_state_labels,
                save_path=str(save_path)
            )
            assert save_path.exists()


# =============================================================================
# plot_component_weights Tests
# =============================================================================

class TestPlotComponentWeights:
    """Tests for plot_component_weights method"""

    def test_returns_figure(self, fitted_dpca, feature_labels):
        """Test that method returns a matplotlib Figure"""
        viz = DPCAVisualizer()
        fig = viz.plot_component_weights(
            fitted_dpca, 'condition',
            component=0,
            feature_labels=feature_labels
        )
        assert isinstance(fig, plt.Figure)

    def test_different_components(self, fitted_dpca, feature_labels):
        """Test plotting different components"""
        viz = DPCAVisualizer()

        for comp in range(3):
            fig = viz.plot_component_weights(
                fitted_dpca, 'time',
                component=comp,
                feature_labels=feature_labels
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_different_marginalizations(self, fitted_dpca, feature_labels):
        """Test plotting for different marginalizations"""
        viz = DPCAVisualizer()

        for margin in ['time', 'condition', 'interaction']:
            fig = viz.plot_component_weights(
                fitted_dpca, margin,
                component=0,
                feature_labels=feature_labels
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_without_feature_labels(self, fitted_dpca):
        """Test that default labels are generated when not provided"""
        viz = DPCAVisualizer()
        fig = viz.plot_component_weights(fitted_dpca, 'condition', component=0)
        assert isinstance(fig, plt.Figure)


# =============================================================================
# plot_mental_state_separation Tests
# =============================================================================

class TestPlotMentalStateSeparation:
    """Tests for plot_mental_state_separation method"""

    def test_returns_figure(self, fitted_dpca, sample_data, mental_state_labels):
        """Test that method returns a matplotlib Figure"""
        viz = DPCAVisualizer()
        fig = viz.plot_mental_state_separation(
            fitted_dpca, sample_data, mental_state_labels
        )
        assert isinstance(fig, plt.Figure)

    def test_custom_components(self, fitted_dpca, sample_data, mental_state_labels):
        """Test with custom component selection"""
        viz = DPCAVisualizer()
        fig = viz.plot_mental_state_separation(
            fitted_dpca, sample_data, mental_state_labels,
            components=(0, 2)
        )
        assert isinstance(fig, plt.Figure)

    def test_different_marginalization(self, fitted_dpca, sample_data, mental_state_labels):
        """Test with different marginalization"""
        viz = DPCAVisualizer()
        fig = viz.plot_mental_state_separation(
            fitted_dpca, sample_data, mental_state_labels,
            marginalization='time'
        )
        assert isinstance(fig, plt.Figure)

    def test_save_path(self, fitted_dpca, sample_data, mental_state_labels):
        """Test saving figure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = DPCAVisualizer()
            save_path = Path(tmpdir) / 'separation.png'
            fig = viz.plot_mental_state_separation(
                fitted_dpca, sample_data, mental_state_labels,
                save_path=str(save_path)
            )
            assert save_path.exists()


# =============================================================================
# plot_comprehensive_summary Tests
# =============================================================================

class TestPlotComprehensiveSummary:
    """Tests for plot_comprehensive_summary method"""

    def test_returns_figure(self, fitted_dpca, sample_data, mental_state_labels):
        """Test that method returns a matplotlib Figure"""
        viz = DPCAVisualizer()
        fig = viz.plot_comprehensive_summary(
            fitted_dpca, sample_data, mental_state_labels
        )
        assert isinstance(fig, plt.Figure)

    def test_has_multiple_subplots(self, fitted_dpca, sample_data, mental_state_labels):
        """Test that comprehensive summary has multiple subplots"""
        viz = DPCAVisualizer()
        fig = viz.plot_comprehensive_summary(
            fitted_dpca, sample_data, mental_state_labels
        )
        # Should have 9 subplots (3x3 grid)
        assert len(fig.axes) >= 6  # At least 6 subplots

    def test_with_feature_labels(self, fitted_dpca, sample_data, mental_state_labels, feature_labels):
        """Test with feature labels provided"""
        viz = DPCAVisualizer()
        fig = viz.plot_comprehensive_summary(
            fitted_dpca, sample_data, mental_state_labels,
            feature_labels=feature_labels
        )
        assert isinstance(fig, plt.Figure)

    def test_with_time_axis(self, fitted_dpca, sample_data, mental_state_labels, time_axis):
        """Test with custom time axis"""
        viz = DPCAVisualizer()
        fig = viz.plot_comprehensive_summary(
            fitted_dpca, sample_data, mental_state_labels,
            time_axis=time_axis
        )
        assert isinstance(fig, plt.Figure)

    def test_save_path(self, fitted_dpca, sample_data, mental_state_labels):
        """Test saving figure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = DPCAVisualizer()
            save_path = Path(tmpdir) / 'summary.png'
            fig = viz.plot_comprehensive_summary(
                fitted_dpca, sample_data, mental_state_labels,
                save_path=str(save_path)
            )
            assert save_path.exists()
            # Summary plot should be larger
            assert save_path.stat().st_size > 10000


# =============================================================================
# plot_reconstruction_quality Tests
# =============================================================================

class TestPlotReconstructionQuality:
    """Tests for plot_reconstruction_quality method"""

    def test_returns_figure(self, fitted_dpca, sample_data):
        """Test that method returns a matplotlib Figure"""
        viz = DPCAVisualizer()
        fig = viz.plot_reconstruction_quality(fitted_dpca, sample_data)
        assert isinstance(fig, plt.Figure)

    def test_has_two_subplots(self, fitted_dpca, sample_data):
        """Test that figure has two subplots"""
        viz = DPCAVisualizer()
        fig = viz.plot_reconstruction_quality(fitted_dpca, sample_data)
        assert len(fig.axes) == 2

    def test_custom_feature_and_condition(self, fitted_dpca, sample_data):
        """Test with custom feature and condition indices"""
        viz = DPCAVisualizer()
        fig = viz.plot_reconstruction_quality(
            fitted_dpca, sample_data,
            feature_idx=5,
            condition_idx=2
        )
        assert isinstance(fig, plt.Figure)

    def test_with_time_axis(self, fitted_dpca, sample_data, time_axis):
        """Test with custom time axis"""
        viz = DPCAVisualizer()
        fig = viz.plot_reconstruction_quality(
            fitted_dpca, sample_data,
            time_axis=time_axis
        )
        assert isinstance(fig, plt.Figure)


# =============================================================================
# create_interactive_plot Tests
# =============================================================================

class TestCreateInteractivePlot:
    """Tests for create_interactive_plot method"""

    def test_returns_plotly_figure(self, fitted_dpca, sample_data, mental_state_labels):
        """Test that method returns a plotly Figure (if plotly installed)"""
        viz = DPCAVisualizer()
        try:
            import plotly.graph_objects as go
            fig = viz.create_interactive_plot(
                fitted_dpca, sample_data, mental_state_labels
            )
            assert isinstance(fig, go.Figure)
        except ImportError:
            # Plotly not installed, test that ImportError is raised
            with pytest.raises(ImportError, match="Plotly is required"):
                viz.create_interactive_plot(
                    fitted_dpca, sample_data, mental_state_labels
                )

    def test_with_feature_labels(self, fitted_dpca, sample_data, mental_state_labels, feature_labels):
        """Test with feature labels"""
        viz = DPCAVisualizer()
        try:
            import plotly.graph_objects as go
            fig = viz.create_interactive_plot(
                fitted_dpca, sample_data, mental_state_labels,
                feature_labels=feature_labels
            )
            assert isinstance(fig, go.Figure)
        except ImportError:
            pytest.skip("Plotly not installed")


# =============================================================================
# quick_plot_dpca_results Tests
# =============================================================================

class TestQuickPlotDpcaResults:
    """Tests for quick_plot_dpca_results function"""

    def test_returns_list_of_figures(self, fitted_dpca, sample_data, mental_state_labels):
        """Test that function returns a list of figures"""
        figures = quick_plot_dpca_results(
            fitted_dpca, sample_data, mental_state_labels
        )
        assert isinstance(figures, list)
        assert len(figures) >= 3
        for fig in figures:
            assert isinstance(fig, plt.Figure)

    def test_with_feature_labels(self, fitted_dpca, sample_data, mental_state_labels, feature_labels):
        """Test with feature labels"""
        figures = quick_plot_dpca_results(
            fitted_dpca, sample_data, mental_state_labels,
            feature_labels=feature_labels
        )
        assert len(figures) >= 3

    def test_save_dir_creates_files(self, fitted_dpca, sample_data, mental_state_labels):
        """Test that save_dir creates output files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            figures = quick_plot_dpca_results(
                fitted_dpca, sample_data, mental_state_labels,
                save_dir=tmpdir
            )

            # Check that files were created
            saved_files = list(Path(tmpdir).glob('*.png'))
            assert len(saved_files) >= 3

    def test_creates_save_directory(self, fitted_dpca, sample_data, mental_state_labels):
        """Test that function creates save directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / 'new_output_dir'
            figures = quick_plot_dpca_results(
                fitted_dpca, sample_data, mental_state_labels,
                save_dir=str(new_dir)
            )
            assert new_dir.exists()


# =============================================================================
# GaitDPCA Visualization Tests
# =============================================================================

class TestGaitDPCAVisualization:
    """Tests for visualizing GaitDPCA results"""

    @pytest.fixture
    def fitted_gait_dpca(self):
        """Create fitted GaitDPCA model"""
        np.random.seed(42)
        data = np.random.randn(10, 50, 3)
        model = GaitDPCA(n_components=3)
        model.fit_with_labels(
            data,
            mental_state_labels=['low', 'medium', 'high'],
            feature_labels=[f'f{i}' for i in range(10)]
        )
        return model, data

    def test_visualize_gait_dpca(self, fitted_gait_dpca):
        """Test that GaitDPCA works with visualizer"""
        model, data = fitted_gait_dpca
        viz = DPCAVisualizer()

        # Should work with mental_state marginalization
        fig = viz.plot_explained_variance(model)
        assert isinstance(fig, plt.Figure)

    def test_mental_state_separation_gait_dpca(self, fitted_gait_dpca):
        """Test mental state separation for GaitDPCA"""
        model, data = fitted_gait_dpca
        viz = DPCAVisualizer()

        # GaitDPCA uses 'condition' marginalization for mental states
        # Check which marginalization is available
        available_margins = list(model.components_.keys())
        margin_to_use = 'mental_state' if 'mental_state' in available_margins else 'condition'

        fig = viz.plot_mental_state_separation(
            model, data, ['low', 'medium', 'high'],
            marginalization=margin_to_use
        )
        assert isinstance(fig, plt.Figure)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestVisualizationEdgeCases:
    """Tests for visualization edge cases"""

    def test_single_component(self):
        """Test visualization with single component model"""
        np.random.seed(42)
        data = np.random.randn(10, 50, 4)
        dpca = DemixedPCA(n_components=1)
        dpca.fit(data)

        viz = DPCAVisualizer()
        fig = viz.plot_explained_variance(dpca)
        assert isinstance(fig, plt.Figure)

    def test_two_conditions(self):
        """Test visualization with only 2 conditions"""
        np.random.seed(42)
        data = np.random.randn(10, 50, 2)
        dpca = DemixedPCA(n_components=3)
        dpca.fit(data)

        viz = DPCAVisualizer()
        fig = viz.plot_mental_state_separation(
            dpca, data, ['condition_a', 'condition_b']
        )
        assert isinstance(fig, plt.Figure)

    def test_many_conditions(self):
        """Test visualization with many conditions"""
        np.random.seed(42)
        data = np.random.randn(10, 50, 8)
        dpca = DemixedPCA(n_components=3)
        dpca.fit(data)

        labels = [f'cond_{i}' for i in range(8)]
        viz = DPCAVisualizer()
        fig = viz.plot_mental_state_separation(dpca, data, labels)
        assert isinstance(fig, plt.Figure)

    def test_short_time_series(self):
        """Test visualization with short time series"""
        np.random.seed(42)
        data = np.random.randn(10, 10, 4)  # Only 10 timepoints
        dpca = DemixedPCA(n_components=3)
        dpca.fit(data)

        viz = DPCAVisualizer()
        fig = viz.plot_component_timecourse(
            dpca, data, 'condition',
            components=[0],
            mental_state_labels=['a', 'b', 'c', 'd']
        )
        assert isinstance(fig, plt.Figure)


def run_quick_tests():
    """Run quick visualization tests without pytest"""
    print("=" * 60)
    print("Running Quick Visualization Tests")
    print("=" * 60)

    # Use non-interactive backend
    matplotlib.use('Agg')
    np.random.seed(42)

    # Test 1: DPCAVisualizer initialization
    print("\n[1] Testing DPCAVisualizer initialization...")
    try:
        viz = DPCAVisualizer()
        assert viz.figsize == (12, 8)
        print("    ✓ Initialization successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Test 2: plot_explained_variance
    print("\n[2] Testing plot_explained_variance...")
    try:
        data = np.random.randn(10, 50, 4)
        dpca = DemixedPCA(n_components=3)
        dpca.fit(data)

        viz = DPCAVisualizer()
        fig = viz.plot_explained_variance(dpca)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        print("    ✓ plot_explained_variance successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Test 3: plot_component_timecourse
    print("\n[3] Testing plot_component_timecourse...")
    try:
        fig = viz.plot_component_timecourse(
            dpca, data, 'condition',
            components=[0, 1],
            mental_state_labels=['a', 'b', 'c', 'd']
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        print("    ✓ plot_component_timecourse successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Test 4: plot_comprehensive_summary
    print("\n[4] Testing plot_comprehensive_summary...")
    try:
        fig = viz.plot_comprehensive_summary(
            dpca, data, ['neutral', 'anxious', 'relaxed', 'focused']
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        print("    ✓ plot_comprehensive_summary successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Test 5: quick_plot_dpca_results
    print("\n[5] Testing quick_plot_dpca_results...")
    try:
        figures = quick_plot_dpca_results(
            dpca, data, ['neutral', 'anxious', 'relaxed', 'focused']
        )
        assert len(figures) >= 3
        for fig in figures:
            plt.close(fig)
        print(f"    ✓ quick_plot_dpca_results created {len(figures)} figures")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Test 6: Save to file
    print("\n[6] Testing save to file...")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_plot.png'
            fig = viz.plot_explained_variance(dpca, save_path=str(save_path))
            assert save_path.exists()
            plt.close(fig)
            print(f"    ✓ Saved figure ({save_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("Quick Visualization Tests Complete!")
    print("=" * 60)


if __name__ == '__main__':
    run_quick_tests()
