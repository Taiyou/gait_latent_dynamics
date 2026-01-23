"""
Test suite for data_loader module

Tests for:
- generate_synthetic_gait_data()
- GaitDataLoader class
- create_gait_cycle_time_axis()
- align_to_gait_cycle()
"""

import sys
from pathlib import Path
import tempfile
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    generate_synthetic_gait_data,
    GaitDataLoader,
    create_gait_cycle_time_axis,
    align_to_gait_cycle
)


class TestGenerateSyntheticGaitData:
    """Tests for generate_synthetic_gait_data function"""

    def test_output_shape(self):
        """Test that output has correct shape"""
        n_trials = 30
        n_features = 10
        n_timepoints = 100
        n_conditions = 4

        data, metadata = generate_synthetic_gait_data(
            n_trials=n_trials,
            n_features=n_features,
            n_timepoints=n_timepoints,
            n_conditions=n_conditions
        )

        assert data.shape == (n_trials, n_features, n_timepoints, n_conditions)

    def test_metadata_contents(self):
        """Test that metadata contains expected keys"""
        data, metadata = generate_synthetic_gait_data(
            n_trials=10,
            n_features=5,
            n_timepoints=50,
            n_conditions=3
        )

        expected_keys = [
            'n_trials', 'n_features', 'n_timepoints', 'n_conditions',
            'feature_labels', 'mental_state_labels', 'time_points',
            'generation_params'
        ]

        for key in expected_keys:
            assert key in metadata, f"Missing key: {key}"

        assert metadata['n_trials'] == 10
        assert metadata['n_features'] == 5
        assert metadata['n_timepoints'] == 50
        assert metadata['n_conditions'] == 3
        assert len(metadata['feature_labels']) == 5
        assert len(metadata['mental_state_labels']) == 3
        assert len(metadata['time_points']) == 50

    def test_reproducibility_with_random_state(self):
        """Test that same random_state produces identical data"""
        data1, _ = generate_synthetic_gait_data(
            n_trials=10, n_features=5, n_timepoints=50, n_conditions=3,
            random_state=42
        )

        data2, _ = generate_synthetic_gait_data(
            n_trials=10, n_features=5, n_timepoints=50, n_conditions=3,
            random_state=42
        )

        np.testing.assert_array_equal(data1, data2)

    def test_different_random_states_differ(self):
        """Test that different random_states produce different data"""
        data1, _ = generate_synthetic_gait_data(
            n_trials=10, n_features=5, n_timepoints=50, n_conditions=3,
            random_state=42
        )

        data2, _ = generate_synthetic_gait_data(
            n_trials=10, n_features=5, n_timepoints=50, n_conditions=3,
            random_state=123
        )

        assert not np.allclose(data1, data2)

    def test_effect_strengths(self):
        """Test that effect strengths influence variance"""
        # High mental state effect
        data_high, _ = generate_synthetic_gait_data(
            n_trials=50, n_features=5, n_timepoints=50, n_conditions=4,
            mental_state_effect_strength=1.0,
            noise_level=0.01,
            random_state=42
        )

        # Low mental state effect
        data_low, _ = generate_synthetic_gait_data(
            n_trials=50, n_features=5, n_timepoints=50, n_conditions=4,
            mental_state_effect_strength=0.01,
            noise_level=0.01,
            random_state=42
        )

        # Variance across conditions should be higher with stronger effect
        var_high = np.var(data_high.mean(axis=(0, 2)), axis=1).mean()
        var_low = np.var(data_low.mean(axis=(0, 2)), axis=1).mean()

        assert var_high > var_low

    def test_noise_level(self):
        """Test that noise_level affects data variability"""
        data_noisy, _ = generate_synthetic_gait_data(
            n_trials=50, n_features=5, n_timepoints=50, n_conditions=3,
            noise_level=1.0,
            random_state=42
        )

        data_clean, _ = generate_synthetic_gait_data(
            n_trials=50, n_features=5, n_timepoints=50, n_conditions=3,
            noise_level=0.001,
            random_state=42
        )

        # Trial-to-trial variance should be higher with more noise
        var_noisy = np.var(data_noisy, axis=0).mean()
        var_clean = np.var(data_clean, axis=0).mean()

        assert var_noisy > var_clean

    def test_feature_labels_match_count(self):
        """Test that feature labels match n_features"""
        for n_features in [3, 10, 15]:
            _, metadata = generate_synthetic_gait_data(
                n_trials=5, n_features=n_features, n_timepoints=20, n_conditions=2
            )
            assert len(metadata['feature_labels']) == n_features

    def test_mental_state_labels_match_count(self):
        """Test that mental state labels match n_conditions"""
        for n_conditions in [2, 5, 7]:
            _, metadata = generate_synthetic_gait_data(
                n_trials=5, n_features=5, n_timepoints=20, n_conditions=n_conditions
            )
            assert len(metadata['mental_state_labels']) == n_conditions

    def test_data_is_finite(self):
        """Test that generated data contains no NaN or Inf values"""
        data, _ = generate_synthetic_gait_data(
            n_trials=50, n_features=15, n_timepoints=100, n_conditions=5,
            random_state=42
        )

        assert np.all(np.isfinite(data))


class TestGaitDataLoader:
    """Tests for GaitDataLoader class"""

    def test_initialization_without_dir(self):
        """Test initialization without data directory"""
        loader = GaitDataLoader()
        assert loader.data_dir is None
        assert loader.data is None
        assert loader.metadata == {}

    def test_initialization_with_dir(self):
        """Test initialization with data directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = GaitDataLoader(data_dir=tmpdir)
            assert loader.data_dir == Path(tmpdir)

    def test_load_from_numpy_npy(self):
        """Test loading from .npy file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            test_data = np.random.randn(10, 5, 50, 3)
            filepath = Path(tmpdir) / "test_data.npy"
            np.save(filepath, test_data)

            # Load data
            loader = GaitDataLoader()
            loaded_data, metadata = loader.load_from_numpy(str(filepath))

            np.testing.assert_array_equal(loaded_data, test_data)

    def test_load_from_numpy_npz(self):
        """Test loading from .npz file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            test_data = np.random.randn(10, 5, 50, 3)
            test_metadata = {'n_trials': 10, 'n_features': 5}

            filepath = Path(tmpdir) / "test_data.npz"
            np.savez(filepath, data=test_data, metadata=test_metadata)

            # Load data
            loader = GaitDataLoader()
            loaded_data, metadata = loader.load_from_numpy(str(filepath))

            np.testing.assert_array_equal(loaded_data, test_data)

    def test_load_from_numpy_with_metadata_json(self):
        """Test loading with separate JSON metadata file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data and metadata
            test_data = np.random.randn(10, 5, 50, 3)
            test_metadata = {'n_trials': 10, 'feature_labels': ['f1', 'f2', 'f3', 'f4', 'f5']}

            data_path = Path(tmpdir) / "test_data.npy"
            meta_path = Path(tmpdir) / "test_metadata.json"

            np.save(data_path, test_data)
            with open(meta_path, 'w') as f:
                json.dump(test_metadata, f)

            # Load data
            loader = GaitDataLoader()
            loaded_data, metadata = loader.load_from_numpy(
                str(data_path),
                metadata_path=str(meta_path)
            )

            np.testing.assert_array_equal(loaded_data, test_data)
            assert metadata['n_trials'] == 10
            assert metadata['feature_labels'] == ['f1', 'f2', 'f3', 'f4', 'f5']

    def test_load_from_csv(self):
        """Test loading from CSV file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV data
            rows = []
            for trial in range(3):
                for cond in range(2):
                    for t in range(5):
                        rows.append({
                            'trial': trial,
                            'condition': f'cond_{cond}',
                            'time': t,
                            'feature_a': np.random.randn(),
                            'feature_b': np.random.randn()
                        })

            df = pd.DataFrame(rows)
            filepath = Path(tmpdir) / "test_data.csv"
            df.to_csv(filepath, index=False)

            # Load data
            loader = GaitDataLoader()
            loaded_data, metadata = loader.load_from_csv(
                str(filepath),
                time_column='time',
                condition_column='condition',
                trial_column='trial'
            )

            assert loaded_data.shape == (3, 2, 5, 2)  # trials, features, timepoints, conditions
            assert metadata['n_trials'] == 3
            assert metadata['n_features'] == 2
            assert metadata['n_timepoints'] == 5
            assert metadata['n_conditions'] == 2

    def test_preprocess_normalize(self):
        """Test normalization in preprocessing"""
        loader = GaitDataLoader()

        # Create data with known mean and std
        data = np.random.randn(10, 5, 50, 3) * 10 + 100  # mean ~100, std ~10

        processed = loader.preprocess(data, normalize=True, filter_cutoff=None, resample_points=None)

        # After z-score normalization, mean should be ~0 and std ~1
        assert np.abs(processed.mean()) < 0.1
        assert np.abs(processed.std() - 1.0) < 0.1

    def test_preprocess_no_normalize(self):
        """Test preprocessing without normalization"""
        loader = GaitDataLoader()

        data = np.random.randn(10, 5, 50, 3) * 10 + 100

        processed = loader.preprocess(data, normalize=False, filter_cutoff=None, resample_points=None)

        # Data should be unchanged
        np.testing.assert_array_equal(processed, data)

    def test_preprocess_resample(self):
        """Test resampling in preprocessing"""
        loader = GaitDataLoader()

        data = np.random.randn(10, 5, 50, 3)
        new_points = 100

        processed = loader.preprocess(
            data,
            normalize=False,
            filter_cutoff=None,
            resample_points=new_points
        )

        assert processed.shape == (10, 5, new_points, 3)

    def test_preprocess_filter(self):
        """Test low-pass filtering in preprocessing"""
        loader = GaitDataLoader()

        # Create data with high-frequency noise
        t = np.linspace(0, 2*np.pi, 100)
        signal = np.sin(t)  # Low frequency
        noise = 0.5 * np.sin(20 * t)  # High frequency

        data = np.zeros((1, 1, 100, 1))
        data[0, 0, :, 0] = signal + noise

        processed = loader.preprocess(
            data,
            normalize=False,
            filter_cutoff=0.1,  # Low cutoff to remove high frequency
            resample_points=None
        )

        # Filtered data should be smoother (lower variance of differences)
        diff_original = np.diff(data[0, 0, :, 0])
        diff_filtered = np.diff(processed[0, 0, :, 0])

        assert np.var(diff_filtered) < np.var(diff_original)

    def test_get_trial_averaged_data(self):
        """Test trial averaging"""
        loader = GaitDataLoader()

        data = np.random.randn(20, 5, 50, 3)

        averaged = loader.get_trial_averaged_data(data)

        assert averaged.shape == (5, 50, 3)
        np.testing.assert_array_almost_equal(averaged, data.mean(axis=0))

    def test_get_trial_averaged_data_no_data(self):
        """Test trial averaging with no data raises error"""
        loader = GaitDataLoader()

        with pytest.raises(ValueError, match="No data available"):
            loader.get_trial_averaged_data()

    def test_save_npy(self):
        """Test saving to .npy format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = GaitDataLoader()
            data = np.random.randn(10, 5, 50, 3)
            metadata = {'n_trials': 10}

            filepath = Path(tmpdir) / "saved_data.npy"
            loader.save(data, str(filepath), metadata=metadata)

            # Verify files exist
            assert filepath.exists()
            assert filepath.with_suffix('.json').exists()

            # Verify data
            loaded = np.load(filepath)
            np.testing.assert_array_equal(loaded, data)

    def test_save_npz(self):
        """Test saving to .npz format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = GaitDataLoader()
            data = np.random.randn(10, 5, 50, 3)
            metadata = {'n_trials': 10}

            filepath = Path(tmpdir) / "saved_data.npz"
            loader.save(data, str(filepath), metadata=metadata)

            # Verify file exists
            assert filepath.exists()

            # Verify data
            loaded = np.load(filepath, allow_pickle=True)
            np.testing.assert_array_equal(loaded['data'], data)

    def test_save_unsupported_format(self):
        """Test saving with unsupported format raises error"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = GaitDataLoader()
            data = np.random.randn(10, 5, 50, 3)

            filepath = Path(tmpdir) / "saved_data.txt"

            with pytest.raises(ValueError, match="Unsupported file format"):
                loader.save(data, str(filepath))

    def test_data_property(self):
        """Test data property returns loaded data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = GaitDataLoader()

            # Initially None
            assert loader.data is None

            # After loading
            test_data = np.random.randn(10, 5, 50, 3)
            filepath = Path(tmpdir) / "test.npy"
            np.save(filepath, test_data)
            loader.load_from_numpy(str(filepath))

            np.testing.assert_array_equal(loader.data, test_data)

    def test_metadata_property(self):
        """Test metadata property returns metadata"""
        loader = GaitDataLoader()

        # Initially empty
        assert loader.metadata == {}


class TestCreateGaitCycleTimeAxis:
    """Tests for create_gait_cycle_time_axis function"""

    def test_default_length(self):
        """Test default 100 points"""
        time_axis = create_gait_cycle_time_axis()
        assert len(time_axis) == 100

    def test_custom_length(self):
        """Test custom number of points"""
        for n_points in [50, 100, 200]:
            time_axis = create_gait_cycle_time_axis(n_points)
            assert len(time_axis) == n_points

    def test_range(self):
        """Test that time axis spans 0 to 100"""
        time_axis = create_gait_cycle_time_axis(100)
        assert time_axis[0] == 0
        assert time_axis[-1] == 100

    def test_evenly_spaced(self):
        """Test that points are evenly spaced"""
        time_axis = create_gait_cycle_time_axis(101)
        diffs = np.diff(time_axis)
        np.testing.assert_array_almost_equal(diffs, np.ones(100))


class TestAlignToGaitCycle:
    """Tests for align_to_gait_cycle function"""

    def test_output_shape(self):
        """Test output shape is correct"""
        n_features = 5
        n_samples = 500
        target_length = 100

        data = np.random.randn(n_features, n_samples)
        heel_strikes = [0, 100, 200, 300, 400]  # 4 cycles

        cycles = align_to_gait_cycle(data, heel_strikes, target_length)

        assert cycles.shape == (4, n_features, target_length)

    def test_interpolation_preserves_endpoints(self):
        """Test that cycle start/end values are preserved"""
        n_features = 2
        target_length = 50

        # Create simple linear data
        data = np.arange(200).reshape(1, 200).repeat(n_features, axis=0).astype(float)
        heel_strikes = [0, 100, 200]

        cycles = align_to_gait_cycle(data, heel_strikes, target_length)

        # First point of first cycle should be close to 0
        assert np.abs(cycles[0, 0, 0] - 0) < 1

        # Last point of first cycle should be close to 99
        assert np.abs(cycles[0, 0, -1] - 99) < 2

    def test_single_cycle(self):
        """Test with single gait cycle"""
        data = np.random.randn(3, 100)
        heel_strikes = [0, 100]

        cycles = align_to_gait_cycle(data, heel_strikes, target_length=50)

        assert cycles.shape == (1, 3, 50)

    def test_variable_cycle_lengths(self):
        """Test with variable length cycles"""
        data = np.random.randn(2, 350)
        # Cycles of different lengths
        heel_strikes = [0, 80, 180, 300]  # lengths: 80, 100, 120

        cycles = align_to_gait_cycle(data, heel_strikes, target_length=100)

        # All cycles should be normalized to target length
        assert cycles.shape == (3, 2, 100)


def run_quick_tests():
    """Run quick tests without pytest"""
    print("=" * 60)
    print("Running Quick Data Loader Tests")
    print("=" * 60)

    np.random.seed(42)

    # Test 1: generate_synthetic_gait_data
    print("\n[1] Testing generate_synthetic_gait_data...")
    try:
        data, metadata = generate_synthetic_gait_data(
            n_trials=10, n_features=5, n_timepoints=50, n_conditions=3,
            random_state=42
        )
        assert data.shape == (10, 5, 50, 3)
        assert len(metadata['feature_labels']) == 5
        print(f"    ✓ Shape: {data.shape}, metadata keys: {list(metadata.keys())}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Test 2: GaitDataLoader save/load
    print("\n[2] Testing GaitDataLoader save/load...")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = GaitDataLoader()
            test_data = np.random.randn(10, 5, 50, 3)
            filepath = Path(tmpdir) / "test.npz"
            loader.save(test_data, str(filepath), metadata={'test': True})
            loaded, meta = loader.load_from_numpy(str(filepath))
            assert np.allclose(loaded, test_data)
            print("    ✓ Save/load roundtrip successful")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Test 3: preprocess
    print("\n[3] Testing preprocess...")
    try:
        loader = GaitDataLoader()
        data = np.random.randn(10, 5, 50, 3) * 10 + 100
        processed = loader.preprocess(data, normalize=True)
        assert np.abs(processed.mean()) < 0.1
        print(f"    ✓ Normalized mean: {processed.mean():.4f}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Test 4: create_gait_cycle_time_axis
    print("\n[4] Testing create_gait_cycle_time_axis...")
    try:
        time_axis = create_gait_cycle_time_axis(100)
        assert time_axis[0] == 0 and time_axis[-1] == 100
        print(f"    ✓ Time axis range: {time_axis[0]} to {time_axis[-1]}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    # Test 5: align_to_gait_cycle
    print("\n[5] Testing align_to_gait_cycle...")
    try:
        data = np.random.randn(3, 300)
        heel_strikes = [0, 100, 200, 300]
        cycles = align_to_gait_cycle(data, heel_strikes, target_length=50)
        assert cycles.shape == (3, 3, 50)
        print(f"    ✓ Aligned cycles shape: {cycles.shape}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("Quick Data Loader Tests Complete!")
    print("=" * 60)


if __name__ == '__main__':
    run_quick_tests()
