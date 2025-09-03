"""
Simple Comprehensive Analysis without matplotlib issues.

This module provides a streamlined analysis focusing on the core results
without complex plotting that might cause issues.

Author: Research Team
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import json
from typing import Dict, Tuple, Any, Optional, List
import time
from datetime import datetime

# Core imports
from ground_truth_model import create_standard_ground_truth
from nonlinear_loudspeaker_model import NonlinearLoudspeakerModel
from phase3_demo import (
    fast_bayesian_identification_method,
    gp_surrogate_identification_method
)


class SimpleResultsAnalyzer:
    """
    Simple comprehensive analysis of all system identification methods.
    """
    
    def __init__(self, results_dir: str = "results"):
        """Initialize results analyzer."""
        self.results_dir = results_dir
        self.ground_truth = create_standard_ground_truth()
        self.results = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/data", exist_ok=True)
    
    def generate_complex_tone_signal(self, n_samples: int = 2000, dt: float = 1e-4,
                                   f1: float = 15.0, f2: float = 600.0,
                                   amplitude1: float = 0.5, amplitude2: float = 0.3) -> Dict[str, jnp.ndarray]:
        """Generate complex tone signal with 15Hz and 600Hz components."""
        t = jnp.linspace(0, (n_samples - 1) * dt, n_samples)
        
        # Complex tone: 15Hz + 600Hz
        tone1 = amplitude1 * jnp.sin(2 * jnp.pi * f1 * t)  # 15 Hz
        tone2 = amplitude2 * jnp.sin(2 * jnp.pi * f2 * t)  # 600 Hz
        
        # Add some phase variation for realism
        phase_shift = 0.1 * jnp.sin(2 * jnp.pi * 0.5 * t)
        u = tone1 + tone2 + phase_shift
        
        # Generate ground truth response
        x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
        t_sim, x_traj = self.ground_truth.simulate(u, x0, dt)
        y_true = jnp.array([self.ground_truth.output(x, u[i]) for i, x in enumerate(x_traj)])
        
        # Add realistic noise
        noise_level = 0.01
        noise = jax.random.normal(jax.random.PRNGKey(42), y_true.shape) * noise_level
        y_measured = y_true + noise
        
        return {
            'u': u,
            'y_true': y_true,
            'y_measured': y_measured,
            't': t,
            'frequencies': [f1, f2],
            'amplitudes': [amplitude1, amplitude2],
            'signal_type': 'complex_tone'
        }
    
    def generate_pink_noise_signal(self, n_samples: int = 2000, dt: float = 1e-4) -> Dict[str, jnp.ndarray]:
        """Generate pink noise signal."""
        t = jnp.linspace(0, (n_samples - 1) * dt, n_samples)
        
        # Generate pink noise (1/f noise)
        key = jax.random.PRNGKey(123)
        white_noise = jax.random.normal(key, (n_samples,))
        
        # Simple pink noise approximation
        pink_noise = jnp.cumsum(white_noise) * 0.1
        
        # Normalize and scale
        u = pink_noise / jnp.std(pink_noise) * 0.5
        
        # Generate ground truth response
        x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
        t_sim, x_traj = self.ground_truth.simulate(u, x0, dt)
        y_true = jnp.array([self.ground_truth.output(x, u[i]) for i, x in enumerate(x_traj)])
        
        # Add realistic noise
        noise_level = 0.01
        noise = jax.random.normal(jax.random.PRNGKey(456), y_true.shape) * noise_level
        y_measured = y_true + noise
        
        return {
            'u': u,
            'y_true': y_true,
            'y_measured': y_measured,
            't': t,
            'signal_type': 'pink_noise'
        }
    
    def run_all_methods(self, test_data: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        """Run all system identification methods."""
        print("🔬 Running all system identification methods...")
        
        u = test_data['u']
        y = test_data['y_measured']
        results = {}
        
        # 1. Baseline Nonlinear Model
        print("  1. Baseline Nonlinear Model...")
        start_time = time.time()
        baseline_model = NonlinearLoudspeakerModel()
        y_baseline = baseline_model.predict(u)
        baseline_time = time.time() - start_time
        
        results['baseline'] = {
            'predictions': y_baseline,
            'parameters': {
                'Re': float(baseline_model.Re),
                'Le': float(baseline_model.Le),
                'Bl': float(baseline_model.Bl),
                'M': float(baseline_model.M),
                'K': float(baseline_model.K),
                'Rm': float(baseline_model.Rm),
                'L20': float(baseline_model.L20),
                'R20': float(baseline_model.R20)
            },
            'execution_time': baseline_time,
            'method_name': 'Baseline Nonlinear'
        }
        
        # 2. Fast Bayesian Inference
        print("  2. Fast Bayesian Inference...")
        start_time = time.time()
        bayesian_result = fast_bayesian_identification_method(u, y, num_samples=15)
        bayesian_time = time.time() - start_time
        
        if bayesian_result:
            results['bayesian'] = {
                'predictions': bayesian_result['predictions'],
                'parameters': {k: float(v) if isinstance(v, (int, float, jnp.ndarray)) else str(v) 
                             for k, v in bayesian_result['parameters'].items()},
                'execution_time': bayesian_time,
                'method_name': 'Fast Bayesian',
                'posterior_stats': bayesian_result.get('posterior_stats', {})
            }
        else:
            print("    ❌ Bayesian method failed")
        
        # 3. GP Surrogate
        print("  3. GP Surrogate Modeling...")
        start_time = time.time()
        gp_result = gp_surrogate_identification_method(u, y)
        gp_time = time.time() - start_time
        
        if gp_result:
            results['gp_surrogate'] = {
                'predictions': gp_result['predictions'],
                'parameters': gp_result.get('training_results', {}),
                'execution_time': gp_time,
                'method_name': 'GP Surrogate'
            }
        else:
            print("    ❌ GP surrogate method failed")
        
        return results
    
    def calculate_comprehensive_metrics(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        error = y_true - y_pred
        
        # Basic metrics
        mse = jnp.mean(error ** 2)
        rmse = jnp.sqrt(mse)
        mae = jnp.mean(jnp.abs(error))
        
        # R² calculation
        ss_res = jnp.sum((y_true - y_pred) ** 2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Normalized metrics
        y_range = jnp.max(y_true) - jnp.min(y_true)
        nrmse = rmse / y_range if y_range > 0 else 0
        
        # Correlation
        correlation = jnp.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'nrmse': float(nrmse),
            'correlation': float(correlation)
        }
    
    def save_comprehensive_results(self, all_results: Dict[str, Dict[str, Any]]):
        """Save comprehensive results to JSON."""
        print("💾 Saving comprehensive results...")
        
        # Prepare results for JSON serialization
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'comprehensive_comparison',
            'signals': {}
        }
        
        for signal_name, signal_results in all_results.items():
            json_results['signals'][signal_name] = {
                'signal_type': signal_name,
                'methods': {}
            }
            
            for method_key, method_result in signal_results.items():
                if 'predictions' in method_result:
                    # Calculate metrics
                    test_data = self.get_test_data(signal_name)
                    metrics = self.calculate_comprehensive_metrics(
                        test_data['y_measured'], method_result['predictions']
                    )
                    
                    # Calculate error timeseries
                    error_timeseries = test_data['y_measured'] - method_result['predictions']
                    
                    json_results['signals'][signal_name]['methods'][method_key] = {
                        'method_name': method_result['method_name'],
                        'metrics': metrics,
                        'execution_time': method_result['execution_time'],
                        'parameters': method_result['parameters'],
                        'error_timeseries': {
                            'current': np.array(error_timeseries[:, 0]).tolist(),
                            'velocity': np.array(error_timeseries[:, 1]).tolist(),
                            'time': np.array(test_data['t']).tolist()
                        },
                        'predictions': {
                            'current': np.array(method_result['predictions'][:, 0]).tolist(),
                            'velocity': np.array(method_result['predictions'][:, 1]).tolist()
                        },
                        'ground_truth': {
                            'current': np.array(test_data['y_measured'][:, 0]).tolist(),
                            'velocity': np.array(test_data['y_measured'][:, 1]).tolist()
                        }
                    }
        
        # Save to JSON
        with open(f'{self.results_dir}/comprehensive_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"  ✅ Results saved to {self.results_dir}/comprehensive_results.json")
    
    def create_summary_report(self, all_results: Dict[str, Dict[str, Any]]):
        """Create a text summary report."""
        print("📝 Creating summary report...")
        
        report_lines = []
        report_lines.append("COMPREHENSIVE LOUDSPEAKER SYSTEM IDENTIFICATION RESULTS")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Overall ranking
        all_method_scores = []
        for signal_name, signal_results in all_results.items():
            for method_key, method_result in signal_results.items():
                if 'predictions' in method_result:
                    test_data = self.get_test_data(signal_name)
                    metrics = self.calculate_comprehensive_metrics(
                        test_data['y_measured'], method_result['predictions']
                    )
                    all_method_scores.append({
                        'method': method_result['method_name'],
                        'signal': signal_name,
                        'r2': metrics['r2'],
                        'mse': metrics['mse'],
                        'time': method_result['execution_time']
                    })
        
        # Sort by R²
        all_method_scores.sort(key=lambda x: x['r2'], reverse=True)
        
        report_lines.append("OVERALL METHOD RANKING (by R² score):")
        report_lines.append("-" * 50)
        for i, score in enumerate(all_method_scores, 1):
            report_lines.append(f"{i}. {score['method']:20s} | {score['signal']:15s} | R² = {score['r2']:8.4f}")
        report_lines.append("")
        
        # Detailed results by signal
        for signal_name, signal_results in all_results.items():
            report_lines.append(f"{signal_name.replace('_', ' ').title()} Results:")
            report_lines.append("-" * 50)
            
            for method_key, method_result in signal_results.items():
                if 'predictions' in method_result:
                    test_data = self.get_test_data(signal_name)
                    metrics = self.calculate_comprehensive_metrics(
                        test_data['y_measured'], method_result['predictions']
                    )
                    
                    report_lines.append(f"\n{method_result['method_name']}:")
                    report_lines.append(f"  R² Score: {metrics['r2']:.4f}")
                    report_lines.append(f"  Final Loss (MSE): {metrics['mse']:.6f}")
                    report_lines.append(f"  RMSE: {metrics['rmse']:.6f}")
                    report_lines.append(f"  MAE: {metrics['mae']:.6f}")
                    report_lines.append(f"  NRMSE: {metrics['nrmse']:.4f}")
                    report_lines.append(f"  Correlation: {metrics['correlation']:.4f}")
                    report_lines.append(f"  Execution Time: {method_result['execution_time']:.2f}s")
                    report_lines.append(f"  Model Parameters: {method_result['parameters']}")
            
            report_lines.append("")
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(f'{self.results_dir}/comprehensive_summary.txt', 'w') as f:
            f.write(report_text)
        
        print(f"  ✅ Summary report saved to {self.results_dir}/comprehensive_summary.txt")
        
        # Print summary to console
        print("\n" + report_text)
    
    def get_test_data(self, signal_name: str) -> Dict[str, jnp.ndarray]:
        """Get test data for a specific signal."""
        if signal_name == 'complex_tone':
            return self.generate_complex_tone_signal()
        elif signal_name == 'pink_noise':
            return self.generate_pink_noise_signal()
        else:
            raise ValueError(f"Unknown signal: {signal_name}")
    
    def run_comprehensive_analysis(self):
        """Run complete comprehensive analysis."""
        print("🚀 COMPREHENSIVE RESULTS ANALYSIS")
        print("=" * 80)
        
        # Generate test signals
        print("\n📊 Generating test signals...")
        complex_tone_data = self.generate_complex_tone_signal()
        pink_noise_data = self.generate_pink_noise_signal()
        
        print(f"  ✅ Complex tone signal: {len(complex_tone_data['u'])} samples")
        print(f"  ✅ Pink noise signal: {len(pink_noise_data['u'])} samples")
        print(f"  ✅ Complex tone frequencies: {complex_tone_data['frequencies']} Hz")
        
        # Run all methods on both signals
        all_results = {}
        
        print("\n🔬 Running methods on complex tone signal (15Hz + 600Hz)...")
        all_results['complex_tone'] = self.run_all_methods(complex_tone_data)
        
        print("\n🔬 Running methods on pink noise signal...")
        all_results['pink_noise'] = self.run_all_methods(pink_noise_data)
        
        # Save results
        self.save_comprehensive_results(all_results)
        
        # Create summary report
        self.create_summary_report(all_results)
        
        print(f"\n✅ All results saved to {self.results_dir}/")
        print("✅ Complex tone (15Hz + 600Hz) analysis completed")
        print("✅ Pink noise analysis completed")
        print("✅ R² comparison across all methods")
        print("✅ Error timeseries analysis completed")
        
        return all_results


def main():
    """Run comprehensive results analysis."""
    # Create analyzer
    analyzer = SimpleResultsAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
    print("=" * 80)
    print("All methods compared on complex tone (15Hz + 600Hz) and pink noise")
    print("R² comparison across all methods and signals")
    print("Error timeseries analysis for all combinations")
    print("Results saved to 'results/' directory")
    print("=" * 80)


if __name__ == "__main__":
    main()
