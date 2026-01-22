"""
Test suite for dPCA and dPLS implementations
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest
from src.dpca import DemixedPCA, DemixedPLS, GaitDPCA, ContinuousScoreDPCA, MultiVariateMentalDPCA


class TestDemixedPCA:
    """Tests for DemixedPCA class"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
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
    
    def test_init(self):
        """Test initialization"""
        dpca = DemixedPCA(n_components=5)
        assert dpca.n_components == 5
        assert dpca.regularizer == 'auto'
    
    def test_fit(self, sample_data):
        """Test fitting"""
        dpca = DemixedPCA(n_components=3)
        dpca.fit(sample_data)
        
        assert 'time' in dpca.components_
        assert 'condition' in dpca.components_
        assert 'interaction' in dpca.components_
        assert dpca.mean_ is not None
    
    def test_transform(self, sample_data):
        """Test transform"""
        dpca = DemixedPCA(n_components=3)
        dpca.fit(sample_data)
        
        # Transform all marginalizations
        Z = dpca.transform(sample_data)
        assert isinstance(Z, dict)
        assert 'time' in Z
        assert 'condition' in Z
        
        # Transform single marginalization
        Z_cond = dpca.transform(sample_data, marginalization='condition')
        assert Z_cond.shape[0] == 3  # n_components
        assert Z_cond.shape[1] == 50  # n_timepoints
        assert Z_cond.shape[2] == 4  # n_conditions
    
    def test_inverse_transform(self, sample_data):
        """Test inverse transform"""
        dpca = DemixedPCA(n_components=3)
        dpca.fit(sample_data)
        
        Z = dpca.transform(sample_data)
        X_reconstructed = dpca.inverse_transform(Z)
        
        assert X_reconstructed.shape == sample_data.shape
    
    def test_explained_variance(self, sample_data):
        """Test explained variance"""
        dpca = DemixedPCA(n_components=3)
        dpca.fit(sample_data)
        
        assert 'time' in dpca.explained_variance_
        assert len(dpca.explained_variance_['time']) == 3


class TestDemixedPLS:
    """Tests for DemixedPLS class"""
    
    @pytest.fixture
    def sample_data_with_y(self):
        """Generate sample data with target"""
        np.random.seed(42)
        n_features = 10
        n_timepoints = 50
        n_conditions = 4
        
        # Mental scores per condition
        Y = np.array([3.0, 5.0, 7.0, 4.0])
        
        t = np.linspace(0, 2*np.pi, n_timepoints)
        X = np.zeros((n_features, n_timepoints, n_conditions))
        
        for f in range(n_features):
            for c in range(n_conditions):
                # Features 0, 1 correlate with Y
                if f < 2:
                    effect = 0.5 * (Y[c] - Y.mean())
                else:
                    effect = 0.1 * (c - 1.5)
                X[f, :, c] = np.sin(t + f*0.3) + effect + np.random.randn(n_timepoints)*0.1
        
        return X, Y
    
    def test_init(self):
        """Test initialization"""
        dpls = DemixedPLS(n_components=3)
        assert dpls.n_components == 3
        assert dpls.scale_y == True
    
    def test_fit(self, sample_data_with_y):
        """Test fitting"""
        X, Y = sample_data_with_y
        dpls = DemixedPLS(n_components=3)
        dpls.fit(X, Y)
        
        assert 'condition' in dpls.components_
        assert dpls.y_mean_ is not None
        assert dpls.mean_ is not None
    
    def test_transform(self, sample_data_with_y):
        """Test transform"""
        X, Y = sample_data_with_y
        dpls = DemixedPLS(n_components=3)
        dpls.fit(X, Y)
        
        Z = dpls.transform(X)
        assert isinstance(Z, dict)
        
        Z_cond = dpls.transform(X, marginalization='condition')
        assert Z_cond.ndim == 3
    
    def test_predict(self, sample_data_with_y):
        """Test prediction"""
        X, Y = sample_data_with_y
        dpls = DemixedPLS(n_components=3)
        dpls.fit(X, Y)
        
        Y_pred = dpls.predict(X)
        assert Y_pred.shape == Y.shape
    
    def test_feature_importance(self, sample_data_with_y):
        """Test feature importance"""
        X, Y = sample_data_with_y
        dpls = DemixedPLS(n_components=3)
        dpls.fit(X, Y, feature_labels=[f'f{i}' for i in range(10)])
        
        importance = dpls.get_feature_importance('condition')
        assert len(importance) == 10
        assert all(v >= 0 for v in importance.values())


class TestContinuousScoreDPCA:
    """Tests for ContinuousScoreDPCA class"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data"""
        np.random.seed(42)
        n_subjects = 20
        n_features = 8
        n_timepoints = 50
        
        # Wellbeing scores
        scores = np.random.uniform(3, 8, n_subjects)
        
        # Gait data
        t = np.linspace(0, 2*np.pi, n_timepoints)
        X = np.zeros((n_subjects, n_features, n_timepoints))
        
        for i in range(n_subjects):
            for f in range(n_features):
                base = np.sin(t + f*0.3)
                if f < 3:
                    effect = 0.3 * (scores[i] - 5)
                else:
                    effect = 0
                X[i, f, :] = base + effect + np.random.randn(n_timepoints)*0.1
        
        return X, scores
    
    def test_fit(self, sample_data):
        """Test fitting"""
        X, scores = sample_data
        model = ContinuousScoreDPCA(n_components=3)
        model.fit(X, scores)
        
        assert model.score_weights_ is not None
        assert len(model.score_weights_) == 8  # n_features
    
    def test_predict(self, sample_data):
        """Test prediction"""
        X, scores = sample_data
        model = ContinuousScoreDPCA(n_components=3)
        model.fit(X, scores)
        
        predicted = model.predict_score(X)
        assert predicted.shape == scores.shape
        
        # Check correlation
        corr = np.corrcoef(scores, predicted)[0, 1]
        assert corr > 0.5  # Should have reasonable correlation


class TestMultiVariateMentalDPCA:
    """Tests for MultiVariateMentalDPCA class"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data"""
        np.random.seed(42)
        n_subjects = 25
        n_timepoints = 50
        n_body_factors = 10
        n_mental_vars = 4
        
        # Mental scores [n_subjects, n_mental_vars]
        mental = np.random.randn(n_subjects, n_mental_vars)
        
        # Gait data [n_subjects, n_timepoints, n_body_factors]
        t = np.linspace(0, 2*np.pi, n_timepoints)
        gait = np.zeros((n_subjects, n_timepoints, n_body_factors))
        
        for i in range(n_subjects):
            for f in range(n_body_factors):
                base = np.sin(t + f*0.3)
                effect = 0.3 * mental[i, f % n_mental_vars]
                gait[i, :, f] = base + effect + np.random.randn(n_timepoints)*0.1
        
        return gait, mental
    
    def test_fit(self, sample_data):
        """Test fitting"""
        gait, mental = sample_data
        model = MultiVariateMentalDPCA(n_gait_components=5, method='cca')
        model.fit(gait, mental)
        
        assert model.gait_latent_ is not None
        assert model.correlations_ is not None
    
    def test_associations(self, sample_data):
        """Test getting associations"""
        gait, mental = sample_data
        model = MultiVariateMentalDPCA(n_gait_components=5, method='pls')
        model.fit(
            gait, mental,
            mental_labels=['var1', 'var2', 'var3', 'var4'],
            gait_labels=[f'gait{i}' for i in range(10)]
        )
        
        associations = model.get_mental_gait_associations()
        assert len(associations) == 4  # n_mental_vars


class TestGaitDPCA:
    """Tests for GaitDPCA class"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data"""
        np.random.seed(42)
        n_features = 10
        n_timepoints = 50
        n_conditions = 3
        
        t = np.linspace(0, 2*np.pi, n_timepoints)
        data = np.zeros((n_features, n_timepoints, n_conditions))
        
        for f in range(n_features):
            for c in range(n_conditions):
                data[f, :, c] = np.sin(t + f*0.3) + 0.5*(c-1) + np.random.randn(n_timepoints)*0.1
        
        return data
    
    def test_fit_with_labels(self, sample_data):
        """Test fitting with labels"""
        model = GaitDPCA(n_components=3)
        model.fit_with_labels(
            sample_data,
            mental_state_labels=['low', 'medium', 'high'],
            feature_labels=[f'f{i}' for i in range(10)]
        )
        
        assert model.components_ is not None
    
    def test_mental_state_separation(self, sample_data):
        """Test mental state separation analysis"""
        model = GaitDPCA(n_components=3)
        model.fit_with_labels(
            sample_data,
            mental_state_labels=['low', 'medium', 'high']
        )
        
        separation = model.analyze_mental_state_separation(
            sample_data, ['low', 'medium', 'high']
        )
        
        assert len(separation) == 3  # 3 pairs


def run_quick_tests():
    """Run quick tests without pytest"""
    print("=" * 60)
    print("Running Quick Tests")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Test 1: DemixedPCA
    print("\n[1] Testing DemixedPCA...")
    try:
        data = np.random.randn(10, 50, 4)
        dpca = DemixedPCA(n_components=3)
        dpca.fit(data)
        Z = dpca.transform(data)
        print("    ✓ DemixedPCA: fit and transform OK")
    except Exception as e:
        print(f"    ✗ DemixedPCA failed: {e}")
    
    # Test 2: DemixedPLS
    print("\n[2] Testing DemixedPLS...")
    try:
        X = np.random.randn(10, 50, 4)
        Y = np.array([3.0, 5.0, 7.0, 4.0])
        dpls = DemixedPLS(n_components=3)
        dpls.fit(X, Y)
        Y_pred = dpls.predict(X)
        print(f"    ✓ DemixedPLS: fit and predict OK (Y_pred shape: {Y_pred.shape})")
    except Exception as e:
        print(f"    ✗ DemixedPLS failed: {e}")
    
    # Test 3: ContinuousScoreDPCA
    print("\n[3] Testing ContinuousScoreDPCA...")
    try:
        X = np.random.randn(20, 8, 50)
        scores = np.random.uniform(3, 8, 20)
        model = ContinuousScoreDPCA(n_components=3)
        model.fit(X, scores)
        predicted = model.predict_score(X)
        corr = np.corrcoef(scores, predicted)[0, 1]
        print(f"    ✓ ContinuousScoreDPCA: OK (correlation: {corr:.3f})")
    except Exception as e:
        print(f"    ✗ ContinuousScoreDPCA failed: {e}")
    
    # Test 4: MultiVariateMentalDPCA
    print("\n[4] Testing MultiVariateMentalDPCA...")
    try:
        gait = np.random.randn(25, 50, 10)
        mental = np.random.randn(25, 4)
        model = MultiVariateMentalDPCA(n_gait_components=5, method='cca')
        model.fit(gait, mental)
        associations = model.get_mental_gait_associations()
        print(f"    ✓ MultiVariateMentalDPCA: OK ({len(associations)} associations)")
    except Exception as e:
        print(f"    ✗ MultiVariateMentalDPCA failed: {e}")
    
    # Test 5: GaitDPCA
    print("\n[5] Testing GaitDPCA...")
    try:
        data = np.random.randn(10, 50, 3)
        model = GaitDPCA(n_components=3)
        model.fit_with_labels(data, mental_state_labels=['low', 'medium', 'high'])
        separation = model.analyze_mental_state_separation(data, ['low', 'medium', 'high'])
        print(f"    ✓ GaitDPCA: OK ({len(separation)} separation scores)")
    except Exception as e:
        print(f"    ✗ GaitDPCA failed: {e}")
    
    print("\n" + "=" * 60)
    print("Quick Tests Complete!")
    print("=" * 60)


if __name__ == '__main__':
    run_quick_tests()
