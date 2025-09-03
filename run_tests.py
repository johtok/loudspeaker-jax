#!/usr/bin/env python3
"""
Quick test runner to verify the comprehensive testing framework.

This script runs a quick test to verify that all components work correctly
and produces the required metrics.

Author: Research Team
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

def test_ground_truth_model():
    """Test ground truth model functionality."""
    print("Testing Ground Truth Model...")
    
    from src.ground_truth_model import create_standard_ground_truth
    
    # Create model
    model = create_standard_ground_truth()
    print(f"  ✓ Created model with {model.n_states} states")
    
    # Test dynamics
    x = jnp.array([0.1, 0.001, 0.01, 0.05])
    u = jnp.array(2.0)
    dxdt = model.dynamics(x, u)
    print(f"  ✓ Dynamics function works: {dxdt.shape}")
    
    # Test output
    y = model.output(x, u)
    print(f"  ✓ Output function works: {y.shape}")
    
    # Test simulation
    u_test = 2.0 * jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 0.1, 100))
    t, x_traj = model.simulate(u_test, jnp.zeros(4), dt=1e-4)
    print(f"  ✓ Simulation works: {x_traj.shape}")
    
    # Test synthetic data generation
    data = model.generate_synthetic_data(u_test, noise_level=0.01)
    print(f"  ✓ Synthetic data generation works: {len(data)} keys")
    
    return True

def test_comprehensive_testing():
    """Test comprehensive testing framework."""
    print("\nTesting Comprehensive Testing Framework...")
    
    from src.ground_truth_model import create_standard_ground_truth
    from src.comprehensive_testing import ComprehensiveTester
    
    # Create tester
    ground_truth = create_standard_ground_truth()
    tester = ComprehensiveTester(ground_truth)
    print("  ✓ Created tester")
    
    # Generate test data
    test_data = tester.generate_test_data(
        excitation_type='sine',
        duration=0.1,
        amplitude=1.0,
        noise_level=0.01
    )
    print(f"  ✓ Generated test data: {len(test_data['voltage'])} samples")
    
    # Test metric calculations
    y_true = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_pred = jnp.array([[1.1, 2.1], [2.1, 3.1], [3.1, 4.1]])
    
    loss = tester._calculate_loss(y_true, y_pred)
    r2 = tester._calculate_r2(y_true, y_pred)
    nrmse = tester._calculate_nrmse(y_true, y_pred)
    mae = tester._calculate_mae(y_true, y_pred)
    correlation = tester._calculate_correlation(y_true, y_pred)
    
    print(f"  ✓ Loss calculation: {loss:.6f}")
    print(f"  ✓ R² calculation: {r2:.4f}")
    print(f"  ✓ NRMSE calculation: {nrmse}")
    print(f"  ✓ MAE calculation: {mae}")
    print(f"  ✓ Correlation calculation: {correlation}")
    
    return True

def test_dynax_identification():
    """Test Dynax identification methods."""
    print("\nTesting Dynax Identification...")
    
    from src.dynax_identification import (
        DynaxLoudspeakerModel, 
        dynax_identification_method
    )
    
    # Test model creation
    model = DynaxLoudspeakerModel()
    print(f"  ✓ Created Dynax model: {model.n_states} states")
    
    # Test dynamics
    x = jnp.array([0.1, 0.001, 0.01, 0.05])
    u = jnp.array(2.0)
    dxdt = model.f(x, u)
    print(f"  ✓ Dynax dynamics: {dxdt.shape}")
    
    # Test output
    y = model.h(x, u)
    print(f"  ✓ Dynax output: {y.shape}")
    
    # Test identification method
    n_samples = 100
    u_test = 2.0 * jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, n_samples))
    y_test = jnp.zeros((n_samples, 2))
    
    result = dynax_identification_method(u_test, y_test, sample_rate=48000)
    print(f"  ✓ Dynax identification: {len(result)} result keys")
    
    return True

def test_comprehensive_framework():
    """Test the complete comprehensive framework."""
    print("\nTesting Complete Framework...")
    
    from src.ground_truth_model import create_standard_ground_truth
    from src.comprehensive_testing import ComprehensiveTester
    from src.dynax_identification import dynax_identification_method
    
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
    
    # Create simple test method
    def simple_method(u: jnp.ndarray, y: jnp.ndarray, **kwargs) -> dict:
        return {
            'model': None,
            'parameters': {'Re': 6.8, 'Le': 0.5e-3, 'Bl': 3.2},
            'predictions': 0.1 * y,  # 10% of true values
            'convergence': {'converged': True}
        }
    
    # Test method
    result = tester.test_method(
        method_func=simple_method,
        method_name='Simple Test',
        framework='Test Framework',
        test_data=test_data
    )
    
    print(f"  ✓ Test method execution: {result.method_name}")
    print(f"  ✓ Final R²: {result.final_r2:.4f}")
    print(f"  ✓ Final loss: {result.final_loss:.6f}")
    print(f"  ✓ Training time: {result.training_time:.2f}s")
    
    # Test comparison
    comparison = tester.compare_methods(test_data)
    print(f"  ✓ Method comparison: {len(comparison.results)} methods")
    print(f"  ✓ Best method: {comparison.best_method}")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("COMPREHENSIVE TESTING FRAMEWORK VERIFICATION")
    print("=" * 60)
    
    try:
        # Test individual components
        test_ground_truth_model()
        test_comprehensive_testing()
        test_dynax_identification()
        test_comprehensive_framework()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe comprehensive testing framework is ready!")
        print("You can now run:")
        print("  python src/run_comprehensive_tests.py --help")
        print("  python src/run_comprehensive_tests.py --ground-truth standard --excitation pink_noise")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
