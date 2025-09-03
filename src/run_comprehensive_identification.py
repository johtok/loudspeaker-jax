"""
Main script to run comprehensive system identification.

This script demonstrates the complete framework with all methods:
- Nonlinear optimization
- Bayesian inference  
- GP surrogates
- State-space modeling
- Original Dynax methods

Author: Research Team
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any
import argparse
from pathlib import Path

# Import our comprehensive framework
from comprehensive_system_identification import (
    ComprehensiveSystemIdentifier,
    run_comprehensive_identification
)
from ground_truth_model import (
    GroundTruthLoudspeakerModel,
    create_standard_ground_truth
)


def generate_test_data(n_samples: int = 1000, dt: float = 1e-4, 
                      noise_level: float = 0.01) -> Dict[str, jnp.ndarray]:
    """
    Generate synthetic test data for system identification.
    
    Args:
        n_samples: Number of samples
        dt: Time step [s]
        noise_level: Noise level for measurements
        
    Returns:
        Dictionary with test data
    """
    print("Generating synthetic test data...")
    
    # Create ground truth model
    ground_truth = create_standard_ground_truth()
    
    # Generate input signal (pink noise + sine waves)
    t = jnp.linspace(0, (n_samples - 1) * dt, n_samples)
    
    # Pink noise component
    pink_noise = jnp.cumsum(jax.random.normal(jax.random.PRNGKey(42), (n_samples,))) * 0.1
    
    # Sine wave components
    sine1 = 0.5 * jnp.sin(2 * jnp.pi * 50 * t)  # 50 Hz
    sine2 = 0.3 * jnp.sin(2 * jnp.pi * 200 * t)  # 200 Hz
    sine3 = 0.2 * jnp.sin(2 * jnp.pi * 1000 * t)  # 1000 Hz
    
    # Combined input
    u = pink_noise + sine1 + sine2 + sine3
    
    # Generate measurements
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])  # Initial state
    t_sim, x_traj = ground_truth.simulate(u, x0, dt)
    
    # Extract outputs (current and velocity)
    y_true = jnp.array([ground_truth.output(x, u[i]) for i, x in enumerate(x_traj)])
    
    # Add noise
    noise = jax.random.normal(jax.random.PRNGKey(123), y_true.shape) * noise_level
    y_measured = y_true + noise
    
    print(f"  Generated {n_samples} samples")
    print(f"  Input range: [{jnp.min(u):.3f}, {jnp.max(u):.3f}] V")
    print(f"  Output range: [{jnp.min(y_measured):.3f}, {jnp.max(y_measured):.3f}]")
    
    return {
        'u': u,
        'y_true': y_true,
        'y_measured': y_measured,
        't': t,
        'ground_truth': ground_truth
    }


def run_identification_experiment(n_samples: int = 1000, 
                                noise_level: float = 0.01,
                                output_dir: str = "comprehensive_results") -> Dict[str, Any]:
    """
    Run comprehensive system identification experiment.
    
    Args:
        n_samples: Number of samples
        noise_level: Noise level for measurements
        output_dir: Output directory for results
        
    Returns:
        Dictionary with experiment results
    """
    print("🔬 COMPREHENSIVE SYSTEM IDENTIFICATION EXPERIMENT")
    print("=" * 80)
    
    # Generate test data
    test_data = generate_test_data(n_samples, noise_level=noise_level)
    u = test_data['u']
    y = test_data['y_measured']
    
    # Create comprehensive identifier
    identifier = ComprehensiveSystemIdentifier()
    identifier.set_ground_truth_model(test_data['ground_truth'])
    
    # Run all methods
    results = identifier.run_all_methods(u, y)
    
    # Generate comprehensive report
    report = identifier.generate_comprehensive_report(output_dir)
    
    return {
        'test_data': test_data,
        'results': results,
        'report': report
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Comprehensive System Identification')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples (default: 1000)')
    parser.add_argument('--noise', type=float, default=0.01,
                       help='Noise level (default: 0.01)')
    parser.add_argument('--output', type=str, default='comprehensive_results',
                       help='Output directory (default: comprehensive_results)')
    parser.add_argument('--methods', nargs='+', 
                       choices=['nonlinear', 'bayesian', 'gp', 'statespace', 'dynax', 'all'],
                       default=['all'],
                       help='Methods to run (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Samples: {args.samples}")
    print(f"  Noise level: {args.noise}")
    print(f"  Output directory: {args.output}")
    print(f"  Methods: {args.methods}")
    print()
    
    # Generate test data
    test_data = generate_test_data(args.samples, noise_level=args.noise)
    u = test_data['u']
    y = test_data['y_measured']
    
    # Create identifier
    identifier = ComprehensiveSystemIdentifier()
    identifier.set_ground_truth_model(test_data['ground_truth'])
    
    # Run selected methods
    all_results = {}
    
    if 'all' in args.methods or 'nonlinear' in args.methods:
        all_results.update(identifier.run_nonlinear_optimization_methods(u, y))
    
    if 'all' in args.methods or 'bayesian' in args.methods:
        all_results.update(identifier.run_bayesian_methods(u, y))
    
    if 'all' in args.methods or 'gp' in args.methods:
        all_results.update(identifier.run_gp_surrogate_methods(u, y))
    
    if 'all' in args.methods or 'statespace' in args.methods:
        all_results.update(identifier.run_state_space_methods(u, y))
    
    if 'all' in args.methods or 'dynax' in args.methods:
        all_results.update(identifier.run_original_dynax_methods(u, y))
    
    # Store results
    identifier.results = {k: v for k, v in all_results.items() if v is not None}
    
    # Generate report
    report = identifier.generate_comprehensive_report(args.output)
    
    print("\n🎉 EXPERIMENT COMPLETED!")
    print(f"Results saved to: {args.output}")
    
    return {
        'test_data': test_data,
        'results': identifier.results,
        'report': report
    }


if __name__ == "__main__":
    main()
