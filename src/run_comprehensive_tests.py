"""
Comprehensive Test Runner for Loudspeaker System Identification.

This script runs comprehensive tests of all system identification methods
and generates detailed reports with the requested metrics:
- Model parameters
- Error timeseries  
- Final loss
- Final R²

Author: Research Team
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any
import argparse
from pathlib import Path
import json
import time

from src.ground_truth_model import (
    create_standard_ground_truth, 
    create_highly_nonlinear_ground_truth,
    create_linear_ground_truth
)
from src.comprehensive_testing import ComprehensiveTester
from src.dynax_identification import (
    dynax_identification_method,
    dynax_linear_only_method
)


def create_placeholder_methods():
    """Create placeholder methods for testing framework."""
    
    def linear_identification_method(u: jnp.ndarray, y: jnp.ndarray, **kwargs) -> Dict[str, Any]:
        """Placeholder linear identification method."""
        print("Running linear identification method...")
        
        # Simple linear model fitting (placeholder)
        n_samples = len(u)
        
        # Estimate basic parameters from data
        Re_est = jnp.std(u) / jnp.std(y[:, 0])  # Rough estimate
        Le_est = 0.5e-3  # Typical value
        Bl_est = 3.0  # Typical value
        M_est = 12e-3  # Typical value
        K_est = 1000  # Typical value
        Rm_est = 0.8  # Typical value
        
        parameters = {
            'Re': float(Re_est),
            'Le': Le_est,
            'Bl': Bl_est,
            'M': M_est,
            'K': K_est,
            'Rm': Rm_est
        }
        
        # Simple prediction (placeholder)
        predictions = jnp.zeros_like(y)
        
        return {
            'model': None,
            'parameters': parameters,
            'predictions': predictions,
            'convergence': {'converged': True, 'iterations': 1}
        }
    
    def nonlinear_identification_method(u: jnp.ndarray, y: jnp.ndarray, **kwargs) -> Dict[str, Any]:
        """Placeholder nonlinear identification method."""
        print("Running nonlinear identification method...")
        
        # Estimate parameters with some nonlinearities
        Re_est = jnp.std(u) / jnp.std(y[:, 0])
        
        parameters = {
            'Re': float(Re_est),
            'Le': 0.5e-3,
            'Bl': 3.2,
            'M': 12e-3,
            'K': 1200,
            'Rm': 0.8,
            'Bl_nl': [0.0, 0.0, -50.0, -0.1],
            'K_nl': [0.0, 0.0, 100.0, 0.0]
        }
        
        # Simple prediction (placeholder)
        predictions = jnp.zeros_like(y)
        
        return {
            'model': None,
            'parameters': parameters,
            'predictions': predictions,
            'convergence': {'converged': True, 'iterations': 10}
        }
    
    def bayesian_identification_method(u: jnp.ndarray, y: jnp.ndarray, **kwargs) -> Dict[str, Any]:
        """Placeholder Bayesian identification method."""
        print("Running Bayesian identification method...")
        
        # Bayesian estimation (placeholder)
        Re_est = jnp.std(u) / jnp.std(y[:, 0])
        
        parameters = {
            'Re': float(Re_est),
            'Le': 0.5e-3,
            'Bl': 3.2,
            'M': 12e-3,
            'K': 1200,
            'Rm': 0.8,
            'Re_std': 0.1,
            'Le_std': 0.05e-3,
            'Bl_std': 0.1
        }
        
        # Simple prediction (placeholder)
        predictions = jnp.zeros_like(y)
        
        return {
            'model': None,
            'parameters': parameters,
            'predictions': predictions,
            'convergence': {'converged': True, 'samples': 1000}
        }
    
    def gp_surrogate_method(u: jnp.ndarray, y: jnp.ndarray, **kwargs) -> Dict[str, Any]:
        """Placeholder GP surrogate method."""
        print("Running GP surrogate method...")
        
        # GP surrogate (placeholder)
        parameters = {
            'kernel_lengthscale': 0.1,
            'kernel_variance': 1.0,
            'noise_variance': 0.01,
            'n_inducing_points': 50
        }
        
        # Simple prediction (placeholder)
        predictions = jnp.zeros_like(y)
        
        return {
            'model': None,
            'parameters': parameters,
            'predictions': predictions,
            'convergence': {'converged': True, 'iterations': 20}
        }
    
    return {
        'Linear Identification': linear_identification_method,
        'Nonlinear Identification': nonlinear_identification_method,
        'Bayesian Identification': bayesian_identification_method,
        'GP Surrogate': gp_surrogate_method,
        'Dynax Full': dynax_identification_method,
        'Dynax Linear': dynax_linear_only_method
    }


def run_comprehensive_tests(ground_truth_type: str = 'standard',
                          excitation_type: str = 'pink_noise',
                          duration: float = 1.0,
                          noise_level: float = 0.01,
                          output_dir: str = 'test_results') -> str:
    """
    Run comprehensive tests of all system identification methods.
    
    Args:
        ground_truth_type: Type of ground truth model ('standard', 'highly_nonlinear', 'linear')
        excitation_type: Type of excitation signal
        duration: Duration of test signal in seconds
        noise_level: Relative noise level
        output_dir: Output directory for results
        
    Returns:
        Path to results directory
    """
    print("=" * 80)
    print("COMPREHENSIVE LOUDSPEAKER SYSTEM IDENTIFICATION TESTING")
    print("=" * 80)
    print(f"Ground Truth Model: {ground_truth_type}")
    print(f"Excitation Type: {excitation_type}")
    print(f"Duration: {duration}s")
    print(f"Noise Level: {noise_level}")
    print("=" * 80)
    
    # Create ground truth model
    if ground_truth_type == 'standard':
        ground_truth = create_standard_ground_truth()
    elif ground_truth_type == 'highly_nonlinear':
        ground_truth = create_highly_nonlinear_ground_truth()
    elif ground_truth_type == 'linear':
        ground_truth = create_linear_ground_truth()
    else:
        raise ValueError(f"Unknown ground truth type: {ground_truth_type}")
    
    print(f"Ground Truth Parameters:")
    for param, value in ground_truth.get_parameters().items():
        print(f"  {param}: {value}")
    print()
    
    # Initialize tester
    tester = ComprehensiveTester(ground_truth)
    
    # Generate test data
    print("Generating test data...")
    test_data = tester.generate_test_data(
        excitation_type=excitation_type,
        duration=duration,
        noise_level=noise_level
    )
    print(f"  Generated {len(test_data['voltage'])} samples")
    print(f"  Sample rate: {test_data['sample_rate']} Hz")
    print(f"  Excitation amplitude: {test_data['amplitude']} V")
    print()
    
    # Get identification methods
    methods = create_placeholder_methods()
    
    # Test each method
    print("Testing identification methods...")
    print("-" * 50)
    
    for method_name, method_func in methods.items():
        # Determine framework
        if 'Dynax' in method_name:
            framework = 'Dynax'
        elif 'Bayesian' in method_name:
            framework = 'NumPyro+BlackJAX'
        elif 'GP' in method_name:
            framework = 'GPJax'
        else:
            framework = 'JAX+Diffrax'
        
        # Run test
        result = tester.test_method(
            method_func=method_func,
            method_name=method_name,
            framework=framework,
            test_data=test_data,
            sample_rate=test_data['sample_rate']
        )
    
    print("-" * 50)
    print()
    
    # Compare methods
    print("Comparing methods...")
    comparison_result = tester.compare_methods(test_data)
    
    print(f"Method Ranking (by R² score):")
    for i, method in enumerate(comparison_result.method_ranking, 1):
        result = next(r for r in comparison_result.results if r.method_name == method)
        print(f"  {i}. {method} ({result.framework}): R² = {result.final_r2:.4f}")
    print()
    
    # Generate comprehensive report
    print("Generating comprehensive report...")
    results_path = tester.generate_report(comparison_result, output_dir)
    
    print(f"Results saved to: {results_path}")
    print()
    
    # Print summary
    print("SUMMARY")
    print("=" * 50)
    print(f"Best Method: {comparison_result.best_method}")
    best_result = next(r for r in comparison_result.results 
                      if r.method_name == comparison_result.best_method)
    print(f"  R² Score: {best_result.final_r2:.4f}")
    print(f"  Final Loss: {best_result.final_loss:.6f}")
    print(f"  Training Time: {best_result.training_time:.2f}s")
    print()
    
    print("All Methods Performance:")
    for result in comparison_result.results:
        print(f"  {result.method_name} ({result.framework}):")
        print(f"    R² = {result.final_r2:.4f}, Loss = {result.final_loss:.6f}, Time = {result.training_time:.2f}s")
    print()
    
    return results_path


def main():
    """Main entry point for comprehensive testing."""
    parser = argparse.ArgumentParser(description="Run comprehensive loudspeaker system identification tests")
    
    parser.add_argument('--ground-truth', choices=['standard', 'highly_nonlinear', 'linear'],
                       default='standard', help='Ground truth model type')
    parser.add_argument('--excitation', choices=['pink_noise', 'sine', 'chirp', 'multitone'],
                       default='pink_noise', help='Excitation signal type')
    parser.add_argument('--duration', type=float, default=1.0,
                       help='Test signal duration in seconds')
    parser.add_argument('--noise-level', type=float, default=0.01,
                       help='Relative noise level')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Run comprehensive tests
    results_path = run_comprehensive_tests(
        ground_truth_type=args.ground_truth,
        excitation_type=args.excitation,
        duration=args.duration,
        noise_level=args.noise_level,
        output_dir=args.output_dir
    )
    
    print(f"Comprehensive testing completed!")
    print(f"Results available in: {results_path}")
    print(f"Check the following files:")
    print(f"  - comprehensive_test_report.json: Detailed results")
    print(f"  - test_summary.txt: Text summary")
    print(f"  - error_timeseries.png: Error plots")
    print(f"  - parameter_comparison.png: Parameter comparison")
    print(f"  - performance_comparison.png: Performance comparison")


if __name__ == "__main__":
    main()
