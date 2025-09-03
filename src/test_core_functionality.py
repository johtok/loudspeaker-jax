"""
Test core functionality of the system identification framework.

This script tests the basic components without complex dependencies.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any
import time

# Import core components
from ground_truth_model import create_standard_ground_truth
from nonlinear_loudspeaker_model import NonlinearLoudspeakerModel, NonlinearSystemIdentifier


def test_ground_truth_model():
    """Test ground truth model functionality."""
    print("Testing Ground Truth Model...")
    
    # Create model
    model = create_standard_ground_truth()
    print("  ✓ Model created")
    
    # Test simulation
    u = jnp.ones(100) * 0.1
    x0 = jnp.zeros(4)
    t, x_traj = model.simulate(u, x0, dt=1e-4)
    print(f"  ✓ Simulation: {len(t)} time points, state shape: {x_traj.shape}")
    
    # Test output
    outputs = jnp.array([model.output(x, u[i]) for i, x in enumerate(x_traj)])
    print(f"  ✓ Output shape: {outputs.shape}")
    
    return model, u, outputs


def test_nonlinear_model():
    """Test nonlinear loudspeaker model."""
    print("\nTesting Nonlinear Loudspeaker Model...")
    
    # Create model
    model = NonlinearLoudspeakerModel()
    print("  ✓ Model created")
    
    # Test simulation
    u = jnp.ones(100) * 0.1
    x0 = jnp.zeros(4)
    t, x_traj = model.simulate(u, x0, dt=1e-4)
    print(f"  ✓ Simulation: {len(t)} time points, state shape: {x_traj.shape}")
    
    # Test output
    outputs = model.predict(u, x0)
    print(f"  ✓ Output shape: {outputs.shape}")
    
    return model, u, outputs


def test_system_identification():
    """Test system identification."""
    print("\nTesting System Identification...")
    
    # Generate test data
    ground_truth = create_standard_ground_truth()
    u = jnp.ones(200) * 0.1
    x0 = jnp.zeros(4)
    t, x_traj = ground_truth.simulate(u, x0, dt=1e-4)
    y_true = jnp.array([ground_truth.output(x, u[i]) for i, x in enumerate(x_traj)])
    
    # Add noise
    noise = jax.random.normal(jax.random.PRNGKey(42), y_true.shape) * 0.01
    y_measured = y_true + noise
    
    print(f"  ✓ Generated test data: {y_measured.shape}")
    
    # Create identifier
    model = NonlinearLoudspeakerModel()
    identifier = NonlinearSystemIdentifier(model)
    print("  ✓ Identifier created")
    
    # Test loss function
    params = model.get_parameters()
    loss = identifier.loss_function(params, u, y_measured, x0)
    print(f"  ✓ Loss function: {loss:.6f}")
    
    return identifier, u, y_measured, x0


def test_optimization():
    """Test optimization methods."""
    print("\nTesting Optimization Methods...")
    
    # Get test data
    identifier, u, y_measured, x0 = test_system_identification()
    
    # Test Gauss-Newton
    print("  Testing Gauss-Newton...")
    start_time = time.time()
    try:
        initial_params = identifier.model.get_parameters()
        result = identifier.gauss_newton_optimization(u, y_measured, initial_params, x0, max_iterations=10)
        training_time = time.time() - start_time
        
        print(f"    ✓ Completed in {training_time:.2f}s")
        print(f"    ✓ Final loss: {result['loss']:.6f}")
        print(f"    ✓ Iterations: {result['iterations']}")
        print(f"    ✓ Converged: {result['converged']}")
        
        return result
        
    except Exception as e:
        print(f"    ❌ Failed: {str(e)}")
        return None


def calculate_metrics(y_true, y_pred):
    """Calculate identification metrics."""
    error_timeseries = y_true - y_pred
    final_loss = jnp.mean(error_timeseries ** 2)
    
    # R² calculation
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    final_r2 = 1 - (ss_res / ss_tot)
    
    return {
        'error_timeseries': error_timeseries,
        'final_loss': float(final_loss),
        'final_r2': float(final_r2)
    }


def main():
    """Main test function."""
    print("🧪 TESTING CORE FUNCTIONALITY")
    print("=" * 50)
    
    # Test ground truth model
    gt_model, u_gt, y_gt = test_ground_truth_model()
    
    # Test nonlinear model
    nl_model, u_nl, y_nl = test_nonlinear_model()
    
    # Test system identification
    identifier, u_id, y_id, x0 = test_system_identification()
    
    # Test optimization
    opt_result = test_optimization()
    
    if opt_result is not None:
        print("\n📊 OPTIMIZATION RESULTS")
        print("=" * 30)
        print(f"Final Loss: {opt_result['loss']:.6f}")
        print(f"Iterations: {opt_result['iterations']}")
        print(f"Converged: {opt_result['converged']}")
        
        # Calculate metrics
        fitted_model = NonlinearLoudspeakerModel()
        fitted_model.set_parameters(opt_result['parameters'])
        y_pred = fitted_model.predict(u_id, x0)
        metrics = calculate_metrics(y_id, y_pred)
        
        print(f"\n🎯 IDENTIFICATION METRICS")
        print("=" * 30)
        print(f"Final Loss: {metrics['final_loss']:.6f}")
        print(f"Final R²: {metrics['final_r2']:.4f}")
        print(f"Error Timeseries Length: {len(metrics['error_timeseries'])} samples")
        
        print(f"\n📋 MODEL PARAMETERS")
        print("=" * 30)
        for key, value in opt_result['parameters'].items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
    
    print("\n✅ CORE FUNCTIONALITY TEST COMPLETED!")


if __name__ == "__main__":
    main()
