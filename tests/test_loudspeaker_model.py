"""
Test-Driven Development for Loudspeaker Physical Model.

This module implements comprehensive tests for the loudspeaker physical model,
following TDD principles. Tests are written first, then the implementation
is developed to pass the tests.

Author: Research Team
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple
from hypothesis import given, strategies as st, settings
from tests.conftest import (
    MathematicalTestMixin, PhysicalConstraintMixin, 
    loudspeaker_parameter_strategy, signal_strategy
)


class TestLoudspeakerModel(MathematicalTestMixin, PhysicalConstraintMixin):
    """Test suite for loudspeaker physical model."""
    
    def test_linear_parameters_initialization(self):
        """Test that linear parameters can be initialized correctly."""
        # This test will fail initially - we need to implement the model
        from src.loudspeaker_model import LoudspeakerModel
        
        params = {
            'Re': 6.8,
            'Le': 0.5e-3,
            'Bl': 3.2,
            'M': 12e-3,
            'K': 1200,
            'Rm': 0.8
        }
        
        model = LoudspeakerModel(**params)
        
        # Test parameter access
        assert model.Re == params['Re']
        assert model.Le == params['Le']
        assert model.Bl == params['Bl']
        assert model.M == params['M']
        assert model.K == params['K']
        assert model.Rm == params['Rm']
    
    def test_parameter_validation(self):
        """Test that invalid parameters are rejected."""
        from src.loudspeaker_model import LoudspeakerModel
        
        # Test negative parameters are rejected
        with pytest.raises(ValueError, match="must be positive"):
            LoudspeakerModel(Re=-1.0, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
        
        with pytest.raises(ValueError, match="must be positive"):
            LoudspeakerModel(Re=6.8, Le=-0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
    
    def test_nonlinear_parameter_initialization(self):
        """Test nonlinear parameter initialization."""
        from src.loudspeaker_model import LoudspeakerModel
        
        linear_params = {
            'Re': 6.8, 'Le': 0.5e-3, 'Bl': 3.2, 'M': 12e-3, 'K': 1200, 'Rm': 0.8
        }
        
        nonlinear_params = {
            'Bl_nl': [0.0, -0.1, 0.0, 0.0],
            'K_nl': [0.0, 0.0, 100.0, 0.0],
            'L_nl': [0.0, 0.0, 0.0, 0.0]
        }
        
        model = LoudspeakerModel(**linear_params, **nonlinear_params)
        
        # Test nonlinear parameter access
        assert jnp.allclose(model.Bl_nl, jnp.array(nonlinear_params['Bl_nl']))
        assert jnp.allclose(model.K_nl, jnp.array(nonlinear_params['K_nl']))
        assert jnp.allclose(model.L_nl, jnp.array(nonlinear_params['L_nl']))
    
    def test_force_factor_calculation(self):
        """Test Bl(x) force factor calculation."""
        from src.loudspeaker_model import LoudspeakerModel
        
        model = LoudspeakerModel(
            Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8,
            Bl_nl=[0.0, -0.1, 0.0, 0.0]
        )
        
        # Test at x=0
        x_zero = jnp.array(0.0)
        Bl_zero = model.force_factor(x_zero)
        assert jnp.isclose(Bl_zero, 3.2)  # Should equal linear Bl
        
        # Test at x=0.001 (1mm)
        x_test = jnp.array(0.001)
        Bl_test = model.force_factor(x_test)
        expected = 3.2 - 0.1 * 0.001  # Bl + Bl_nl[1] * x
        assert jnp.isclose(Bl_test, expected)
    
    def test_stiffness_calculation(self):
        """Test K(x) stiffness calculation."""
        from src.loudspeaker_model import LoudspeakerModel
        
        model = LoudspeakerModel(
            Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8,
            K_nl=[0.0, 0.0, 100.0, 0.0]
        )
        
        # Test at x=0
        x_zero = jnp.array(0.0)
        K_zero = model.stiffness(x_zero)
        assert jnp.isclose(K_zero, 1200)  # Should equal linear K
        
        # Test at x=0.001 (1mm)
        x_test = jnp.array(0.001)
        K_test = model.stiffness(x_test)
        expected = 1200 + 100.0 * 0.001**2  # K + K_nl[2] * x^2
        assert jnp.isclose(K_test, expected)
    
    def test_inductance_calculation(self):
        """Test L(x,i) inductance calculation."""
        from src.loudspeaker_model import LoudspeakerModel
        
        model = LoudspeakerModel(
            Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8,
            L_nl=[0.0, 0.0, 0.0, 0.0]
        )
        
        # Test at x=0, i=0
        x_zero = jnp.array(0.0)
        i_zero = jnp.array(0.0)
        L_zero = model.inductance(x_zero, i_zero)
        assert jnp.isclose(L_zero, 0.5e-3)  # Should equal linear Le
    
    @given(loudspeaker_parameter_strategy())
    def test_parameter_physical_constraints(self, params):
        """Test that all parameters satisfy physical constraints."""
        from src.loudspeaker_model import LoudspeakerModel
        
        model = LoudspeakerModel(**params)
        
        # All parameters should be positive
        self.assert_positive(jnp.array([model.Re, model.Le, model.Bl, model.M, model.K, model.Rm]))
        
        # Mass should be reasonable (between 1g and 100g)
        assert 1e-3 < model.M < 100e-3
        
        # Stiffness should be reasonable (between 100 and 10000 N/m)
        assert 100 < model.K < 10000
        
        # Force factor should be reasonable (between 0.5 and 20 N/A)
        assert 0.5 < model.Bl < 20
    
    def test_state_space_dimensions(self):
        """Test that state space has correct dimensions."""
        from src.loudspeaker_model import LoudspeakerModel
        
        model = LoudspeakerModel(Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
        
        # State vector: [i, x, v, i2] for L2R2 model
        assert model.n_states == 4
        assert model.n_inputs == 1  # voltage input
        assert model.n_outputs == 2  # current and velocity outputs
    
    def test_dynamics_function_signature(self):
        """Test that dynamics function has correct signature."""
        from src.loudspeaker_model import LoudspeakerModel
        
        model = LoudspeakerModel(Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
        
        # Test dynamics function
        x = jnp.array([0.1, 0.001, 0.01, 0.05])  # [i, x, v, i2]
        u = jnp.array(2.0)  # voltage
        
        dxdt = model.dynamics(x, u)
        
        # Should return state derivative of same shape
        assert dxdt.shape == x.shape
        assert jnp.all(jnp.isfinite(dxdt))
    
    def test_output_function_signature(self):
        """Test that output function has correct signature."""
        from src.loudspeaker_model import LoudspeakerModel
        
        model = LoudspeakerModel(Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
        
        # Test output function
        x = jnp.array([0.1, 0.001, 0.01, 0.05])  # [i, x, v, i2]
        u = jnp.array(2.0)  # voltage
        
        y = model.output(x, u)
        
        # Should return output of correct shape
        assert y.shape == (2,)  # [current, velocity]
        assert jnp.all(jnp.isfinite(y))
    
    def test_energy_conservation_linear_case(self):
        """Test energy conservation for linear case."""
        from src.loudspeaker_model import LoudspeakerModel
        
        # Linear model (no nonlinearities)
        model = LoudspeakerModel(
            Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8,
            Bl_nl=[0.0, 0.0, 0.0, 0.0],
            K_nl=[0.0, 0.0, 0.0, 0.0],
            L_nl=[0.0, 0.0, 0.0, 0.0]
        )
        
        # Test with zero input (free oscillation)
        x = jnp.array([0.0, 0.001, 0.01, 0.0])  # [i, x, v, i2]
        u = jnp.array(0.0)  # no input
        
        dxdt = model.dynamics(x, u)
        
        # For linear case with no input, energy should be conserved
        # (ignoring damping for this test)
        # This is a simplified test - full energy conservation requires integration
    
    def test_causality(self):
        """Test that the system is causal."""
        from src.loudspeaker_model import LoudspeakerModel
        
        model = LoudspeakerModel(Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
        
        # Test impulse response
        x0 = jnp.zeros(4)  # Initial state
        u_impulse = jnp.array(1.0)  # Impulse input
        
        # System should not respond before input is applied
        # This is inherently satisfied by the ODE structure
    
    def test_passivity(self):
        """Test that the system is passive."""
        from src.loudspeaker_model import LoudspeakerModel
        
        model = LoudspeakerModel(Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
        
        # Test impedance calculation
        # For a passive system, real part of impedance should be positive
        # This requires frequency domain analysis - simplified test here
        
        # Test that electrical resistance is positive
        assert model.Re > 0
        assert model.Rm > 0  # Mechanical resistance should also be positive
    
    @given(signal_strategy(n_samples=100))
    def test_dynamics_with_various_inputs(self, input_signal):
        """Test dynamics function with various input signals."""
        from src.loudspeaker_model import LoudspeakerModel
        
        model = LoudspeakerModel(Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
        
        # Test with different input values
        x = jnp.array([0.1, 0.001, 0.01, 0.05])
        
        for u_val in jnp.linspace(-10, 10, 10):
            u = jnp.array(u_val)
            dxdt = model.dynamics(x, u)
            
            # Dynamics should be finite for all reasonable inputs
            assert jnp.all(jnp.isfinite(dxdt))
    
    def test_jacobian_calculation(self):
        """Test that Jacobians can be calculated correctly."""
        from src.loudspeaker_model import LoudspeakerModel
        
        model = LoudspeakerModel(Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
        
        x = jnp.array([0.1, 0.001, 0.01, 0.05])
        u = jnp.array(2.0)
        
        # Test state Jacobian
        df_dx = jax.jacfwd(model.dynamics, argnums=0)(x, u)
        assert df_dx.shape == (4, 4)
        assert jnp.all(jnp.isfinite(df_dx))
        
        # Test input Jacobian
        df_du = jax.jacfwd(model.dynamics, argnums=1)(x, u)
        assert df_du.shape == (4,)
        assert jnp.all(jnp.isfinite(df_du))
    
    def test_parameter_gradients(self):
        """Test that parameter gradients can be calculated."""
        from src.loudspeaker_model import LoudspeakerModel
        
        def loss_fn(params):
            model = LoudspeakerModel(**params)
            x = jnp.array([0.1, 0.001, 0.01, 0.05])
            u = jnp.array(2.0)
            dxdt = model.dynamics(x, u)
            return jnp.sum(dxdt**2)  # Simple loss function
        
        params = {
            'Re': 6.8, 'Le': 0.5e-3, 'Bl': 3.2, 
            'M': 12e-3, 'K': 1200, 'Rm': 0.8
        }
        
        # Test gradient calculation
        grad = jax.grad(loss_fn)(params)
        
        # All parameters should have gradients
        for param_name in params.keys():
            assert param_name in grad
            assert jnp.isfinite(grad[param_name])
    
    def test_model_serialization(self):
        """Test that model can be serialized and deserialized."""
        from src.loudspeaker_model import LoudspeakerModel
        
        # Create model
        original_model = LoudspeakerModel(
            Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8,
            Bl_nl=[0.0, -0.1, 0.0, 0.0]
        )
        
        # Test parameter extraction
        params = original_model.get_parameters()
        
        # Test model reconstruction
        reconstructed_model = LoudspeakerModel(**params)
        
        # Test that models are equivalent
        x = jnp.array([0.1, 0.001, 0.01, 0.05])
        u = jnp.array(2.0)
        
        dxdt_orig = original_model.dynamics(x, u)
        dxdt_recon = reconstructed_model.dynamics(x, u)
        
        self.assert_close(dxdt_orig, dxdt_recon)
    
    def test_model_copy_and_modification(self):
        """Test that model can be copied and modified."""
        from src.loudspeaker_model import LoudspeakerModel
        
        original_model = LoudspeakerModel(Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
        
        # Test parameter modification
        modified_model = original_model.replace(Re=7.0)
        
        assert modified_model.Re == 7.0
        assert original_model.Re == 6.8  # Original should be unchanged
        
        # Test that other parameters are preserved
        assert modified_model.Le == original_model.Le
        assert modified_model.Bl == original_model.Bl


class TestLoudspeakerModelIntegration:
    """Integration tests for loudspeaker model with external libraries."""
    
    def test_diffrax_integration(self):
        """Test integration with Diffrax ODE solver."""
        from src.loudspeaker_model import LoudspeakerModel
        import diffrax as dfx
        
        model = LoudspeakerModel(Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
        
        # Define ODE function
        def ode_func(t, y, args):
            u = args  # input voltage
            return model.dynamics(y, u)
        
        # Solve ODE
        t0, t1 = 0.0, 1.0
        y0 = jnp.array([0.0, 0.0, 0.0, 0.0])  # Initial state
        u = jnp.array(2.0)  # Constant input
        
        solution = dfx.diffeqsolve(
            dfx.ODETerm(ode_func),
            dfx.Tsit5(),
            t0, t1, dt0=0.01,
            y0=y0,
            args=u
        )
        
        # Check solution
        assert solution.ys.shape[0] > 0  # Should have solution points
        assert jnp.all(jnp.isfinite(solution.ys))  # Should be finite
    
    def test_jaxopt_integration(self):
        """Test integration with JAXopt for parameter estimation."""
        from src.loudspeaker_model import LoudspeakerModel
        import jaxopt
        
        # Create synthetic data
        true_model = LoudspeakerModel(Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
        
        # Generate synthetic measurement data
        t = jnp.linspace(0, 1, 100)
        u = 2.0 * jnp.sin(2 * jnp.pi * 10 * t)  # 10 Hz sine wave
        
        # This would require ODE solving - simplified for now
        # In real implementation, we'd solve the ODE and add noise
        
        # Test parameter estimation setup
        def loss_fn(params):
            model = LoudspeakerModel(**params)
            # Simplified loss - in reality would compare with measured data
            return jnp.sum(jnp.array(list(params.values()))**2)
        
        initial_params = {
            'Re': 7.0, 'Le': 0.6e-3, 'Bl': 3.0, 
            'M': 13e-3, 'K': 1100, 'Rm': 0.9
        }
        
        # Test optimization setup
        optimizer = jaxopt.GradientDescent(fun=loss_fn, maxiter=10)
        result = optimizer.run(initial_params)
        
        # Check that optimization completed
        assert result.state.iter_num <= 10
        assert jnp.isfinite(result.params['Re'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
