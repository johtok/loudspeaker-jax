"""
Test the comprehensive testing framework.

This module tests the comprehensive testing framework to ensure it works
correctly with the ground truth model and produces the required metrics.

Author: Research Team
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.ground_truth_model import (
    create_standard_ground_truth,
    create_highly_nonlinear_ground_truth,
    create_linear_ground_truth
)
from src.comprehensive_testing import ComprehensiveTester, TestResult
from src.dynax_identification import dynax_identification_method
from tests.conftest import MathematicalTestMixin, PhysicalConstraintMixin


class TestGroundTruthModel(MathematicalTestMixin, PhysicalConstraintMixin):
    """Test the ground truth loudspeaker model."""
    
    def test_standard_ground_truth_creation(self):
        """Test creation of standard ground truth model."""
        model = create_standard_ground_truth()
        
        # Test basic properties
        assert model.n_states == 4
        assert model.n_inputs == 1
        assert model.n_outputs == 2
        
        # Test parameter values
        assert model.Re > 0
        assert model.Le > 0
        assert model.Bl > 0
        assert model.M > 0
        assert model.K > 0
        assert model.Rm > 0
    
    def test_ground_truth_dynamics(self):
        """Test ground truth model dynamics."""
        model = create_standard_ground_truth()
        
        # Test state vector
        x = jnp.array([0.1, 0.001, 0.01, 0.05])  # [i, x, v, i2]
        u = jnp.array(2.0)  # voltage
        
        # Test dynamics function
        dxdt = model.dynamics(x, u)
        
        # Should return state derivative of same shape
        assert dxdt.shape == x.shape
        assert jnp.all(jnp.isfinite(dxdt))
    
    def test_ground_truth_output(self):
        """Test ground truth model output function."""
        model = create_standard_ground_truth()
        
        x = jnp.array([0.1, 0.001, 0.01, 0.05])
        u = jnp.array(2.0)
        
        y = model.output(x, u)
        
        # Should return output of correct shape
        assert y.shape == (2,)  # [current, velocity]
        assert jnp.all(jnp.isfinite(y))
    
    def test_ground_truth_simulation(self):
        """Test ground truth model simulation."""
        model = create_standard_ground_truth()
        
        # Create simple input
        n_samples = 100
        u = 2.0 * jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, n_samples))
        x0 = jnp.zeros(4)
        
        # Simulate
        t, x_traj = model.simulate(u, x0, dt=1e-4)
        
        # Check results
        assert len(t) == n_samples
        assert x_traj.shape == (n_samples, 4)
        assert jnp.all(jnp.isfinite(x_traj))
    
    def test_ground_truth_synthetic_data(self):
        """Test synthetic data generation."""
        model = create_standard_ground_truth()
        
        # Create input
        n_samples = 1000
        u = 2.0 * jnp.sin(2 * jnp.pi * 100 * jnp.linspace(0, 1, n_samples))
        
        # Generate synthetic data
        data = model.generate_synthetic_data(u, noise_level=0.01)
        
        # Check data structure
        required_keys = [
            'time', 'voltage', 'current_clean', 'velocity_clean',
            'current_measured', 'velocity_measured', 'current_noise', 'velocity_noise'
        ]
        
        for key in required_keys:
            assert key in data
            assert len(data[key]) == n_samples
            assert jnp.all(jnp.isfinite(data[key]))
    
    def test_nonlinear_parameters(self):
        """Test nonlinear parameter functions."""
        model = create_standard_ground_truth()
        
        # Test force factor
        x = jnp.array([0.001, 0.002, 0.003])
        Bl = model.force_factor(x)
        assert Bl.shape == x.shape
        assert jnp.all(Bl > 0)  # Should be positive
        
        # Test stiffness
        K = model.stiffness(x)
        assert K.shape == x.shape
        assert jnp.all(K > 0)  # Should be positive
        
        # Test inductance
        i = jnp.array([0.1, 0.2, 0.3])
        L = model.inductance(x, i)
        assert L.shape == x.shape
        assert jnp.all(L > 0)  # Should be positive


class TestComprehensiveTester(MathematicalTestMixin, PhysicalConstraintMixin):
    """Test the comprehensive testing framework."""
    
    @pytest.fixture
    def tester(self):
        """Create tester with standard ground truth model."""
        return ComprehensiveTester(create_standard_ground_truth())
    
    def test_tester_initialization(self, tester):
        """Test tester initialization."""
        assert tester.ground_truth is not None
        assert len(tester.results) == 0
    
    def test_test_data_generation(self, tester):
        """Test test data generation."""
        # Test pink noise generation
        data = tester.generate_test_data(
            excitation_type='pink_noise',
            duration=0.1,
            sample_rate=48000,
            amplitude=2.0,
            noise_level=0.01
        )
        
        # Check data structure
        required_keys = [
            'time', 'voltage', 'current_measured', 'velocity_measured',
            'current_clean', 'velocity_clean', 'excitation_type', 'sample_rate'
        ]
        
        for key in required_keys:
            assert key in data
        
        # Check data properties
        assert data['excitation_type'] == 'pink_noise'
        assert data['sample_rate'] == 48000
        assert data['amplitude'] == 2.0
        assert data['noise_level'] == 0.01
        
        # Check data lengths
        n_samples = int(0.1 * 48000)
        assert len(data['time']) == n_samples
        assert len(data['voltage']) == n_samples
        assert len(data['current_measured']) == n_samples
        assert len(data['velocity_measured']) == n_samples
    
    def test_sine_excitation(self, tester):
        """Test sine wave excitation generation."""
        data = tester.generate_test_data(
            excitation_type='sine',
            duration=0.1,
            amplitude=1.0
        )
        
        assert data['excitation_type'] == 'sine'
        assert data['amplitude'] == 1.0
        
        # Check that it's approximately sinusoidal
        voltage = data['voltage']
        assert jnp.allclose(jnp.max(voltage), 1.0, atol=0.1)
        assert jnp.allclose(jnp.min(voltage), -1.0, atol=0.1)
    
    def test_chirp_excitation(self, tester):
        """Test chirp excitation generation."""
        data = tester.generate_test_data(
            excitation_type='chirp',
            duration=0.1,
            amplitude=1.0
        )
        
        assert data['excitation_type'] == 'chirp'
        assert data['amplitude'] == 1.0
    
    def test_multitone_excitation(self, tester):
        """Test multitone excitation generation."""
        data = tester.generate_test_data(
            excitation_type='multitone',
            duration=0.1,
            amplitude=1.0
        )
        
        assert data['excitation_type'] == 'multitone'
        assert data['amplitude'] == 1.0
    
    def test_metric_calculations(self, tester):
        """Test metric calculation functions."""
        # Create test data
        y_true = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y_pred = jnp.array([[1.1, 2.1], [2.1, 3.1], [3.1, 4.1]])
        
        # Test loss calculation
        loss = tester._calculate_loss(y_true, y_pred)
        assert jnp.isfinite(loss)
        assert loss > 0
        
        # Test R² calculation
        r2 = tester._calculate_r2(y_true, y_pred)
        assert jnp.isfinite(r2)
        assert r2 > 0  # Should be positive for good predictions
        
        # Test NRMSE calculation
        nrmse = tester._calculate_nrmse(y_true, y_pred)
        assert 'current' in nrmse
        assert 'velocity' in nrmse
        assert jnp.isfinite(nrmse['current'])
        assert jnp.isfinite(nrmse['velocity'])
        
        # Test MAE calculation
        mae = tester._calculate_mae(y_true, y_pred)
        assert 'current' in mae
        assert 'velocity' in mae
        assert jnp.isfinite(mae['current'])
        assert jnp.isfinite(mae['velocity'])
        
        # Test correlation calculation
        correlation = tester._calculate_correlation(y_true, y_pred)
        assert 'current' in correlation
        assert 'velocity' in correlation
        assert jnp.isfinite(correlation['current'])
        assert jnp.isfinite(correlation['velocity'])
        assert abs(correlation['current']) <= 1.0
        assert abs(correlation['velocity']) <= 1.0


class TestDynaxIdentification:
    """Test Dynax-based identification methods."""
    
    def test_dynax_model_creation(self):
        """Test Dynax model creation."""
        from src.dynax_identification import DynaxLoudspeakerModel
        
        model = DynaxLoudspeakerModel()
        
        # Test basic properties
        assert model.n_states == 4
        assert model.n_inputs == 1
        assert model.n_outputs == 2
        assert model.out == [0, 2]
    
    def test_dynax_dynamics(self):
        """Test Dynax model dynamics."""
        from src.dynax_identification import DynaxLoudspeakerModel
        
        model = DynaxLoudspeakerModel()
        
        # Test state vector
        x = jnp.array([0.1, 0.001, 0.01, 0.05])
        u = jnp.array(2.0)
        
        # Test dynamics function
        dxdt = model.f(x, u)
        
        assert dxdt.shape == x.shape
        assert jnp.all(jnp.isfinite(dxdt))
    
    def test_dynax_output(self):
        """Test Dynax model output."""
        from src.dynax_identification import DynaxLoudspeakerModel
        
        model = DynaxLoudspeakerModel()
        
        x = jnp.array([0.1, 0.001, 0.01, 0.05])
        u = jnp.array(2.0)
        
        y = model.h(x, u)
        
        assert y.shape == (2,)  # [current, velocity]
        assert jnp.all(jnp.isfinite(y))
    
    def test_dynax_identification_method(self):
        """Test Dynax identification method."""
        # Create test data
        n_samples = 100
        u = 2.0 * jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, n_samples))
        y = jnp.zeros((n_samples, 2))  # Placeholder output
        
        # Test identification method
        result = dynax_identification_method(u, y, sample_rate=48000)
        
        # Check result structure
        required_keys = ['model', 'parameters', 'predictions', 'convergence']
        for key in required_keys:
            assert key in result
        
        # Check predictions shape
        assert result['predictions'].shape == y.shape
        
        # Check parameters
        assert isinstance(result['parameters'], dict)
        
        # Check convergence info
        assert isinstance(result['convergence'], dict)


class TestComprehensiveFramework:
    """Test the complete comprehensive framework."""
    
    def test_end_to_end_testing(self):
        """Test end-to-end comprehensive testing."""
        # Create tester
        ground_truth = create_standard_ground_truth()
        tester = ComprehensiveTester(ground_truth)
        
        # Generate test data
        test_data = tester.generate_test_data(
            excitation_type='sine',
            duration=0.1,
            amplitude=1.0,
            noise_level=0.01
        )
        
        # Create a simple test method
        def simple_test_method(u: jnp.ndarray, y: jnp.ndarray, **kwargs) -> dict:
            """Simple test method that returns zeros."""
            return {
                'model': None,
                'parameters': {'Re': 6.8, 'Le': 0.5e-3, 'Bl': 3.2},
                'predictions': jnp.zeros_like(y),
                'convergence': {'converged': True}
            }
        
        # Test the method
        result = tester.test_method(
            method_func=simple_test_method,
            method_name='Simple Test',
            framework='Test Framework',
            test_data=test_data
        )
        
        # Check result
        assert isinstance(result, TestResult)
        assert result.method_name == 'Simple Test'
        assert result.framework == 'Test Framework'
        assert result.final_r2 == -float('inf')  # Should be -inf for zero predictions
        assert result.training_time >= 0
        
        # Check that result was added to tester
        assert len(tester.results) == 1
        assert tester.results[0] == result
    
    def test_method_comparison(self):
        """Test method comparison functionality."""
        # Create tester
        ground_truth = create_standard_ground_truth()
        tester = ComprehensiveTester(ground_truth)
        
        # Generate test data
        test_data = tester.generate_test_data(
            excitation_type='sine',
            duration=0.1,
            amplitude=1.0
        )
        
        # Create two test methods
        def method1(u: jnp.ndarray, y: jnp.ndarray, **kwargs) -> dict:
            return {
                'model': None,
                'parameters': {'Re': 6.8},
                'predictions': 0.1 * y,  # 10% of true values
                'convergence': {'converged': True}
            }
        
        def method2(u: jnp.ndarray, y: jnp.ndarray, **kwargs) -> dict:
            return {
                'model': None,
                'parameters': {'Re': 7.0},
                'predictions': 0.2 * y,  # 20% of true values
                'convergence': {'converged': True}
            }
        
        # Test both methods
        result1 = tester.test_method(method1, 'Method 1', 'Framework 1', test_data)
        result2 = tester.test_method(method2, 'Method 2', 'Framework 2', test_data)
        
        # Compare methods
        comparison = tester.compare_methods(test_data)
        
        # Check comparison results
        assert len(comparison.results) == 2
        assert len(comparison.method_ranking) == 2
        assert comparison.best_method in ['Method 1', 'Method 2']
        assert 'Method 1' in comparison.performance_summary
        assert 'Method 2' in comparison.performance_summary
    
    def test_report_generation(self):
        """Test report generation functionality."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create tester
            ground_truth = create_standard_ground_truth()
            tester = ComprehensiveTester(ground_truth)
            
            # Generate test data
            test_data = tester.generate_test_data(
                excitation_type='sine',
                duration=0.1,
                amplitude=1.0
            )
            
            # Create test method
            def test_method(u: jnp.ndarray, y: jnp.ndarray, **kwargs) -> dict:
                return {
                    'model': None,
                    'parameters': {'Re': 6.8, 'Le': 0.5e-3},
                    'predictions': 0.1 * y,
                    'convergence': {'converged': True}
                }
            
            # Test method
            tester.test_method(test_method, 'Test Method', 'Test Framework', test_data)
            
            # Generate comparison
            comparison = tester.compare_methods(test_data)
            
            # Generate report
            results_path = tester.generate_report(comparison, temp_dir)
            
            # Check that files were created
            results_dir = Path(results_path)
            assert results_dir.exists()
            assert (results_dir / "comprehensive_test_report.json").exists()
            assert (results_dir / "test_summary.txt").exists()
            assert (results_dir / "error_timeseries.png").exists()
            assert (results_dir / "parameter_comparison.png").exists()
            assert (results_dir / "performance_comparison.png").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
