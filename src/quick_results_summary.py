"""
Quick Results Summary - Complex Tone and Pink Noise Analysis.

This script provides a quick summary of results for complex tone (15Hz + 600Hz)
and pink noise signals across all methods.

Author: Research Team
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import json
from typing import Dict, Any
import time

# Core imports
from ground_truth_model import create_standard_ground_truth
from nonlinear_loudspeaker_model import NonlinearLoudspeakerModel
from phase3_demo import (
    fast_bayesian_identification_method,
    gp_surrogate_identification_method
)


def generate_complex_tone_signal(n_samples: int = 1000, dt: float = 1e-4):
    """Generate complex tone signal with 15Hz and 600Hz components."""
    t = jnp.linspace(0, (n_samples - 1) * dt, n_samples)
    
    # Complex tone: 15Hz + 600Hz
    tone1 = 0.5 * jnp.sin(2 * jnp.pi * 15.0 * t)  # 15 Hz
    tone2 = 0.3 * jnp.sin(2 * jnp.pi * 600.0 * t)  # 600 Hz
    u = tone1 + tone2
    
    # Generate ground truth response
    ground_truth = create_standard_ground_truth()
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    t_sim, x_traj = ground_truth.simulate(u, x0, dt)
    y_true = jnp.array([ground_truth.output(x, u[i]) for i, x in enumerate(x_traj)])
    
    # Add noise
    noise = jax.random.normal(jax.random.PRNGKey(42), y_true.shape) * 0.01
    y_measured = y_true + noise
    
    return u, y_measured, y_true


def generate_pink_noise_signal(n_samples: int = 1000, dt: float = 1e-4):
    """Generate pink noise signal."""
    t = jnp.linspace(0, (n_samples - 1) * dt, n_samples)
    
    # Generate pink noise
    key = jax.random.PRNGKey(123)
    white_noise = jax.random.normal(key, (n_samples,))
    pink_noise = jnp.cumsum(white_noise) * 0.1
    u = pink_noise / jnp.std(pink_noise) * 0.5
    
    # Generate ground truth response
    ground_truth = create_standard_ground_truth()
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    t_sim, x_traj = ground_truth.simulate(u, x0, dt)
    y_true = jnp.array([ground_truth.output(x, u[i]) for i, x in enumerate(x_traj)])
    
    # Add noise
    noise = jax.random.normal(jax.random.PRNGKey(456), y_true.shape) * 0.01
    y_measured = y_true + noise
    
    return u, y_measured, y_true


def calculate_metrics(y_true, y_pred):
    """Calculate R² and MSE metrics."""
    error = y_true - y_pred
    mse = jnp.mean(error ** 2)
    
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return float(mse), float(r2)


def run_quick_analysis():
    """Run quick analysis on both signals."""
    print("🚀 QUICK RESULTS ANALYSIS")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    results = {}
    
    # Test complex tone signal
    print("\n📊 Testing Complex Tone Signal (15Hz + 600Hz)...")
    u_complex, y_complex, y_true_complex = generate_complex_tone_signal()
    
    complex_results = {}
    
    # Baseline method
    print("  1. Baseline Nonlinear Model...")
    start_time = time.time()
    baseline_model = NonlinearLoudspeakerModel()
    y_baseline = baseline_model.predict(u_complex)
    baseline_time = time.time() - start_time
    mse_baseline, r2_baseline = calculate_metrics(y_complex, y_baseline)
    
    complex_results['baseline'] = {
        'method': 'Baseline Nonlinear',
        'r2': r2_baseline,
        'mse': mse_baseline,
        'time': baseline_time,
        'parameters': {
            'Re': float(baseline_model.Re),
            'Bl': float(baseline_model.Bl),
            'K': float(baseline_model.K)
        }
    }
    
    # GP Surrogate method
    print("  2. GP Surrogate Modeling...")
    start_time = time.time()
    gp_result = gp_surrogate_identification_method(u_complex, y_complex)
    gp_time = time.time() - start_time
    
    if gp_result:
        mse_gp, r2_gp = calculate_metrics(y_complex, gp_result['predictions'])
        complex_results['gp_surrogate'] = {
            'method': 'GP Surrogate',
            'r2': r2_gp,
            'mse': mse_gp,
            'time': gp_time,
            'parameters': gp_result.get('training_results', {})
        }
    
    results['complex_tone'] = complex_results
    
    # Test pink noise signal
    print("\n📊 Testing Pink Noise Signal...")
    u_pink, y_pink, y_true_pink = generate_pink_noise_signal()
    
    pink_results = {}
    
    # Baseline method
    print("  1. Baseline Nonlinear Model...")
    start_time = time.time()
    y_baseline_pink = baseline_model.predict(u_pink)
    baseline_time_pink = time.time() - start_time
    mse_baseline_pink, r2_baseline_pink = calculate_metrics(y_pink, y_baseline_pink)
    
    pink_results['baseline'] = {
        'method': 'Baseline Nonlinear',
        'r2': r2_baseline_pink,
        'mse': mse_baseline_pink,
        'time': baseline_time_pink,
        'parameters': {
            'Re': float(baseline_model.Re),
            'Bl': float(baseline_model.Bl),
            'K': float(baseline_model.K)
        }
    }
    
    # GP Surrogate method
    print("  2. GP Surrogate Modeling...")
    start_time = time.time()
    gp_result_pink = gp_surrogate_identification_method(u_pink, y_pink)
    gp_time_pink = time.time() - start_time
    
    if gp_result_pink:
        mse_gp_pink, r2_gp_pink = calculate_metrics(y_pink, gp_result_pink['predictions'])
        pink_results['gp_surrogate'] = {
            'method': 'GP Surrogate',
            'r2': r2_gp_pink,
            'mse': mse_gp_pink,
            'time': gp_time_pink,
            'parameters': gp_result_pink.get('training_results', {})
        }
    
    results['pink_noise'] = pink_results
    
    # Create summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n🏆 METHOD RANKING (by R² score):")
    print("-" * 40)
    
    all_scores = []
    for signal_name, signal_results in results.items():
        for method_key, method_result in signal_results.items():
            all_scores.append({
                'method': method_result['method'],
                'signal': signal_name,
                'r2': method_result['r2'],
                'mse': method_result['mse'],
                'time': method_result['time']
            })
    
    all_scores.sort(key=lambda x: x['r2'], reverse=True)
    
    for i, score in enumerate(all_scores, 1):
        print(f"{i}. {score['method']:20s} | {score['signal']:15s} | R² = {score['r2']:8.4f}")
    
    print("\n📊 DETAILED RESULTS:")
    print("-" * 40)
    
    for signal_name, signal_results in results.items():
        print(f"\n{signal_name.replace('_', ' ').title()}:")
        for method_key, method_result in signal_results.items():
            print(f"  {method_result['method']}:")
            print(f"    R² = {method_result['r2']:.4f}")
            print(f"    MSE = {method_result['mse']:.6f}")
            print(f"    Time = {method_result['time']:.2f}s")
    
    # Save results
    with open('results/quick_results_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to results/quick_results_summary.json")
    
    return results


if __name__ == "__main__":
    results = run_quick_analysis()
    
    print("\n🎉 QUICK ANALYSIS COMPLETED!")
    print("=" * 60)
    print("✅ Complex tone (15Hz + 600Hz) analysis completed")
    print("✅ Pink noise analysis completed")
    print("✅ R² comparison across methods")
    print("✅ Results saved to results/ directory")
    print("=" * 60)
