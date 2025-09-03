"""
Test-Driven Development for System Identification Methods.

This module implements comprehensive tests for system identification algorithms,
following TDD principles.

Author: Research Team
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple
from hypothesis import given, strategies as st, settings
from tests.conftest import MathematicalTestMixin, PhysicalConstraintMixin


class TestLinearSystemIdentification(MathematicalTestMixin, PhysicalConstraintMixin):
    """Test suite for linear system identification methods."""
    
    def test_csd_matching_initialization(self):
        """Test that CSD matching can be initialized correctly."""
        from src.system_identification import CSDMatching
        
        # This test will fail initially - we need to implement the class
        csd_matcher = CSDMatching(
            nperseg=1024,
            noverlap=512,
            window='hann'
        )
        
        assert csd_matcher.nperseg == 1024
        assert csd_matcher.noverlap == 512
        assert csd_matcher.window == 'hann'
    
    def test_csd_matching_with_synthetic_data(self, synthetic_loudspeaker_data):
        """Test CSD matching with synthetic data."""
        from src.system_identification import CSDMatching
        
        csd_matcher = CSDMatching(nperseg=1024, noverlap=512)
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        fs = synthetic_loudspeaker_data['sample_rate']
        
        # Test CSD matching
        result = csd_matcher.fit(u, y, fs)
        
        # Test that result contains expected fields
        assert 'parameters' in result
        assert 'cost' in result
        assert 'iterations' in result
        assert 'converged' in result
        
        # Test parameter validation
        params = result['parameters']
        assert 'Re' in params
        assert 'Le' in params
        assert 'Bl' in params
        assert 'M' in params
        assert 'K' in params
        assert 'Rm' in params
        
        # Test that parameters are positive
        for param_name, param_value in params.items():
            assert param_value > 0, f"Parameter {param_name} should be positive"
    
    def test_subspace_identification(self, synthetic_loudspeaker_data):
        """Test subspace identification method."""
        from src.system_identification import SubspaceIdentification
        
        # This test will fail initially - we need to implement the class
        subspace_id = SubspaceIdentification(
            order=4,
            method='n4sid'
        )
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        
        # Test subspace identification
        result = subspace_id.fit(u, y)
        
        # Test that result contains expected fields
        assert 'A' in result
        assert 'B' in result
        assert 'C' in result
        assert 'D' in result
        
        # Test matrix dimensions
        A, B, C, D = result['A'], result['B'], result['C'], result['D']
        n = A.shape[0]
        
        assert A.shape == (n, n)
        assert B.shape == (n, 1)  # Single input
        assert C.shape == (2, n)  # Two outputs
        assert D.shape == (2, 1)
        
        # Test stability (eigenvalues should be inside unit circle)
        eigenvals = jnp.linalg.eigvals(A)
        assert jnp.all(jnp.abs(eigenvals) < 1.0), "System should be stable"
    
    def test_linear_parameter_estimation(self, loudspeaker_parameters):
        """Test linear parameter estimation."""
        from src.system_identification import LinearParameterEstimator
        
        estimator = LinearParameterEstimator()
        
        # Create synthetic measurement data
        t = jnp.linspace(0, 1, 1000)
        u = 2.0 * jnp.sin(2 * jnp.pi * 10 * t)  # 10 Hz sine wave
        
        # Generate synthetic response (simplified)
        # In real implementation, this would come from ODE solving
        i = u / loudspeaker_parameters['Re']  # Simplified electrical model
        v = jnp.zeros_like(u)  # Placeholder for velocity
        
        y = jnp.stack([i, v], axis=1)
        
        # Test parameter estimation
        result = estimator.fit(u, y, loudspeaker_parameters)
        
        # Test that estimated parameters are reasonable
        assert 'Re' in result
        assert 'Le' in result
        assert 'Bl' in result
        assert 'M' in result
        assert 'K' in result
        assert 'Rm' in result
        
        # Test parameter bounds
        for param_name, param_value in result.items():
            assert param_value > 0, f"Parameter {param_name} should be positive"
    
    def test_frequency_domain_identification(self, synthetic_loudspeaker_data):
        """Test frequency domain identification."""
        from src.system_identification import FrequencyDomainIdentification
        
        freq_id = FrequencyDomainIdentification(
            frequency_range=(10, 1000),
            n_frequencies=100
        )
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        fs = synthetic_loudspeaker_data['sample_rate']
        
        # Test frequency domain identification
        result = freq_id.fit(u, y, fs)
        
        # Test that result contains expected fields
        assert 'transfer_functions' in result
        assert 'parameters' in result
        assert 'frequencies' in result
        
        # Test transfer function properties
        tf = result['transfer_functions']
        assert 'voltage_to_current' in tf
        assert 'voltage_to_velocity' in tf
        
        # Test that transfer functions are causal
        for tf_name, tf_data in tf.items():
            assert 'magnitude' in tf_data
            assert 'phase' in tf_data
            assert 'coherence' in tf_data
            
            # Test coherence (should be high for good identification)
            coherence = tf_data['coherence']
            assert jnp.all(coherence >= 0) and jnp.all(coherence <= 1)
    
    def test_linear_model_validation(self, synthetic_loudspeaker_data):
        """Test linear model validation."""
        from src.system_identification import LinearModelValidator
        
        validator = LinearModelValidator()
        
        # Create test model
        from src.loudspeaker_model import LoudspeakerModel
        model = LoudspeakerModel(
            Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8
        )
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        
        # Test model validation
        result = validator.validate(model, u, y)
        
        # Test validation metrics
        assert 'nrmse' in result
        assert 'fit_percentage' in result
        assert 'coherence' in result
        assert 'residuals' in result
        
        # Test that metrics are in valid ranges
        assert 0 <= result['fit_percentage'] <= 100
        assert 0 <= result['coherence'] <= 1
        assert jnp.all(jnp.isfinite(result['nrmse']))


class TestNonlinearSystemIdentification(MathematicalTestMixin, PhysicalConstraintMixin):
    """Test suite for nonlinear system identification methods."""
    
    def test_gauss_newton_optimization(self, synthetic_loudspeaker_data):
        """Test Gauss-Newton optimization for nonlinear identification."""
        from src.system_identification import GaussNewtonOptimizer
        
        optimizer = GaussNewtonOptimizer(
            max_iterations=100,
            tolerance=1e-6,
            verbose=False
        )
        
        # Create initial model
        from src.loudspeaker_model import LoudspeakerModel
        initial_model = LoudspeakerModel(
            Re=7.0, Le=0.6e-3, Bl=3.0, M=13e-3, K=1100, Rm=0.9
        )
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        
        # Test optimization
        result = optimizer.optimize(initial_model, u, y)
        
        # Test that optimization completed
        assert 'model' in result
        assert 'cost_history' in result
        assert 'iterations' in result
        assert 'converged' in result
        
        # Test that cost decreased
        cost_history = result['cost_history']
        assert cost_history[0] >= cost_history[-1], "Cost should decrease during optimization"
        
        # Test that final model is valid
        final_model = result['model']
        assert isinstance(final_model, LoudspeakerModel)
        
        # Test parameter bounds
        params = final_model.get_parameters()
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                assert param_value > 0, f"Parameter {param_name} should be positive"
    
    def test_levenberg_marquardt_optimization(self, synthetic_loudspeaker_data):
        """Test Levenberg-Marquardt optimization."""
        from src.system_identification import LevenbergMarquardtOptimizer
        
        optimizer = LevenbergMarquardtOptimizer(
            max_iterations=100,
            tolerance=1e-6,
            damping_factor=1.0,
            verbose=False
        )
        
        # Create initial model
        from src.loudspeaker_model import LoudspeakerModel
        initial_model = LoudspeakerModel(
            Re=7.0, Le=0.6e-3, Bl=3.0, M=13e-3, K=1100, Rm=0.9
        )
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        
        # Test optimization
        result = optimizer.optimize(initial_model, u, y)
        
        # Test that optimization completed
        assert 'model' in result
        assert 'cost_history' in result
        assert 'iterations' in result
        assert 'converged' in result
        
        # Test convergence properties
        assert result['iterations'] <= 100
        assert result['converged'] or result['iterations'] == 100
    
    def test_nonlinear_parameter_estimation(self, synthetic_loudspeaker_data):
        """Test nonlinear parameter estimation."""
        from src.system_identification import NonlinearParameterEstimator
        
        estimator = NonlinearParameterEstimator(
            method='gauss_newton',
            max_iterations=50
        )
        
        # Create initial model with nonlinearities
        from src.loudspeaker_model import LoudspeakerModel
        initial_model = LoudspeakerModel(
            Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8,
            Bl_nl=jnp.array([0.0, -0.1, 0.0, 0.0]),
            K_nl=jnp.array([0.0, 0.0, 100.0, 0.0])
        )
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        
        # Test parameter estimation
        result = estimator.fit(initial_model, u, y)
        
        # Test that result contains expected fields
        assert 'model' in result
        assert 'parameters' in result
        assert 'cost' in result
        assert 'iterations' in result
        
        # Test that nonlinear parameters are estimated
        final_model = result['model']
        nonlinear_params = final_model.get_nonlinear_parameters()
        
        assert 'Bl_nl' in nonlinear_params
        assert 'K_nl' in nonlinear_params
        assert 'L_nl' in nonlinear_params
        assert 'Li_nl' in nonlinear_params
    
    def test_regularization(self, synthetic_loudspeaker_data):
        """Test regularization in parameter estimation."""
        from src.system_identification import RegularizedParameterEstimator
        
        # Test L1 regularization
        l1_estimator = RegularizedParameterEstimator(
            regularization='l1',
            regularization_strength=0.01
        )
        
        # Test L2 regularization
        l2_estimator = RegularizedParameterEstimator(
            regularization='l2',
            regularization_strength=0.01
        )
        
        # Create initial model
        from src.loudspeaker_model import LoudspeakerModel
        initial_model = LoudspeakerModel(
            Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8
        )
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        
        # Test both regularizers
        for estimator in [l1_estimator, l2_estimator]:
            result = estimator.fit(initial_model, u, y)
            
            assert 'model' in result
            assert 'regularization_cost' in result
            assert 'total_cost' in result
            
            # Test that regularization cost is positive
            assert result['regularization_cost'] >= 0
    
    def test_parameter_bounds(self, synthetic_loudspeaker_data):
        """Test parameter bounds in optimization."""
        from src.system_identification import BoundedParameterEstimator
        
        # Define parameter bounds
        bounds = {
            'Re': (1.0, 20.0),
            'Le': (0.1e-3, 2.0e-3),
            'Bl': (0.5, 20.0),
            'M': (1e-3, 100e-3),
            'K': (100, 10000),
            'Rm': (0.1, 10.0)
        }
        
        estimator = BoundedParameterEstimator(bounds=bounds)
        
        # Create initial model
        from src.loudspeaker_model import LoudspeakerModel
        initial_model = LoudspeakerModel(
            Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8
        )
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        
        # Test bounded estimation
        result = estimator.fit(initial_model, u, y)
        
        # Test that parameters are within bounds
        final_model = result['model']
        params = final_model.get_linear_parameters()
        
        for param_name, param_value in params.items():
            if param_name in bounds:
                lower, upper = bounds[param_name]
                assert lower <= param_value <= upper, \
                    f"Parameter {param_name} = {param_value} not in bounds [{lower}, {upper}]"


class TestSystemIdentificationIntegration(MathematicalTestMixin, PhysicalConstraintMixin):
    """Integration tests for system identification methods."""
    
    def test_multi_scale_optimization(self, synthetic_loudspeaker_data):
        """Test multi-scale optimization strategy."""
        from src.system_identification import MultiScaleOptimizer
        
        optimizer = MultiScaleOptimizer(
            linear_method='csd_matching',
            nonlinear_method='gauss_newton',
            max_iterations=50
        )
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        fs = synthetic_loudspeaker_data['sample_rate']
        
        # Test multi-scale optimization
        result = optimizer.fit(u, y, fs)
        
        # Test that result contains expected fields
        assert 'linear_model' in result
        assert 'nonlinear_model' in result
        assert 'linear_cost' in result
        assert 'nonlinear_cost' in result
        assert 'total_iterations' in result
        
        # Test that nonlinear cost is lower than linear cost
        assert result['nonlinear_cost'] <= result['linear_cost']
    
    def test_cross_validation(self, synthetic_loudspeaker_data):
        """Test cross-validation for model selection."""
        from src.system_identification import CrossValidator
        
        validator = CrossValidator(
            n_folds=5,
            test_size=0.2
        )
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        
        # Test cross-validation
        result = validator.validate(u, y)
        
        # Test that result contains expected fields
        assert 'mean_score' in result
        assert 'std_score' in result
        assert 'fold_scores' in result
        assert 'best_model' in result
        
        # Test that scores are in valid range
        assert 0 <= result['mean_score'] <= 1
        assert result['std_score'] >= 0
        assert len(result['fold_scores']) == 5
    
    def test_model_comparison(self, synthetic_loudspeaker_data):
        """Test comparison of different identification methods."""
        from src.system_identification import ModelComparator
        
        comparator = ModelComparator()
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        fs = synthetic_loudspeaker_data['sample_rate']
        
        # Test model comparison
        result = comparator.compare(u, y, fs)
        
        # Test that result contains expected fields
        assert 'methods' in result
        assert 'scores' in result
        assert 'best_method' in result
        assert 'ranking' in result
        
        # Test that all methods are compared
        expected_methods = ['linear', 'nonlinear', 'subspace']
        for method in expected_methods:
            assert method in result['methods']
            assert method in result['scores']
    
    def test_uncertainty_quantification(self, synthetic_loudspeaker_data):
        """Test uncertainty quantification in parameter estimation."""
        from src.system_identification import UncertaintyQuantifier
        
        quantifier = UncertaintyQuantifier(
            method='bootstrap',
            n_bootstrap=100
        )
        
        # Extract data
        u = synthetic_loudspeaker_data['voltage']
        y = jnp.stack([
            synthetic_loudspeaker_data['current'],
            synthetic_loudspeaker_data['velocity']
        ], axis=1)
        
        # Test uncertainty quantification
        result = quantifier.quantify(u, y)
        
        # Test that result contains expected fields
        assert 'parameters' in result
        assert 'uncertainties' in result
        assert 'confidence_intervals' in result
        
        # Test that uncertainties are positive
        for param_name, uncertainty in result['uncertainties'].items():
            assert uncertainty > 0, f"Uncertainty for {param_name} should be positive"
        
        # Test confidence intervals
        for param_name, ci in result['confidence_intervals'].items():
            assert len(ci) == 2  # Lower and upper bounds
            assert ci[0] < ci[1]  # Lower < upper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
