"""
Phase 4: Comparative Analysis and Benchmarking.

This module implements comprehensive comparative analysis of all system identification methods:
- Statistical evaluation and comparison
- Performance benchmarking
- Robustness testing
- Comprehensive metrics analysis

Author: Research Team
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Core imports
from ground_truth_model import create_standard_ground_truth
from nonlinear_loudspeaker_model import NonlinearLoudspeakerModel, NonlinearSystemIdentifier
from phase3_demo import (
    fast_bayesian_identification_method,
    gp_surrogate_identification_method
)


class ComprehensiveBenchmark:
    """
    Comprehensive benchmarking framework for system identification methods.
    """
    
    def __init__(self):
        """Initialize benchmark framework."""
        self.results = {}
        self.methods = {}
        self.ground_truth = create_standard_ground_truth()
    
    def register_method(self, name: str, method_func: callable, **kwargs):
        """Register a method for benchmarking."""
        self.methods[name] = {
            'function': method_func,
            'kwargs': kwargs
        }
    
    def generate_test_scenarios(self, n_scenarios: int = 10) -> List[Dict[str, Any]]:
        """Generate diverse test scenarios for robust evaluation."""
        scenarios = []
        
        for i in range(n_scenarios):
            # Vary signal characteristics
            n_samples = np.random.randint(200, 1000)
            dt = 1e-4
            
            # Different input types
            input_types = ['pink_noise', 'sine_waves', 'chirp', 'multitone', 'random']
            input_type = np.random.choice(input_types)
            
            # Vary noise levels
            noise_level = np.random.uniform(0.005, 0.05)
            
            # Generate scenario
            scenario = self._generate_scenario(n_samples, dt, input_type, noise_level, seed=i)
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_scenario(self, n_samples: int, dt: float, input_type: str, 
                          noise_level: float, seed: int) -> Dict[str, Any]:
        """Generate a specific test scenario."""
        key = jax.random.PRNGKey(seed)
        t = jnp.linspace(0, (n_samples - 1) * dt, n_samples)
        
        if input_type == 'pink_noise':
            u = jnp.cumsum(jax.random.normal(key, (n_samples,))) * 0.1
        elif input_type == 'sine_waves':
            f1, f2, f3 = np.random.uniform(50, 1000, 3)
            u = (0.5 * jnp.sin(2 * jnp.pi * f1 * t) + 
                 0.3 * jnp.sin(2 * jnp.pi * f2 * t) + 
                 0.2 * jnp.sin(2 * jnp.pi * f3 * t))
        elif input_type == 'chirp':
            f_start, f_end = np.random.uniform(50, 1000, 2)
            u = 0.5 * jnp.sin(2 * jnp.pi * (f_start + (f_end - f_start) * t / t[-1]) * t)
        elif input_type == 'multitone':
            freqs = np.random.uniform(50, 2000, 5)
            u = jnp.sum(jnp.array([0.2 * jnp.sin(2 * jnp.pi * f * t) for f in freqs]), axis=0)
        else:  # random
            u = jax.random.normal(key, (n_samples,)) * 0.5
        
        # Generate ground truth
        x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
        t_sim, x_traj = self.ground_truth.simulate(u, x0, dt)
        y_true = jnp.array([self.ground_truth.output(x, u[i]) for i, x in enumerate(x_traj)])
        
        # Add noise
        noise = jax.random.normal(key, y_true.shape) * noise_level
        y_measured = y_true + noise
        
        return {
            'u': u,
            'y_true': y_true,
            'y_measured': y_measured,
            't': t,
            'input_type': input_type,
            'noise_level': noise_level,
            'n_samples': n_samples,
            'seed': seed
        }
    
    def run_benchmark(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive benchmark across all scenarios and methods."""
        print("🚀 PHASE 4: COMPREHENSIVE BENCHMARKING")
        print("=" * 80)
        
        all_results = {}
        
        for method_name, method_info in self.methods.items():
            print(f"\n📊 Benchmarking {method_name}...")
            method_results = []
            
            for i, scenario in enumerate(scenarios):
                print(f"  Scenario {i+1}/{len(scenarios)}: {scenario['input_type']} "
                      f"(n={scenario['n_samples']}, noise={scenario['noise_level']:.3f})")
                
                try:
                    start_time = time.time()
                    result = method_info['function'](
                        scenario['u'], 
                        scenario['y_measured'],
                        **method_info['kwargs']
                    )
                    execution_time = time.time() - start_time
                    
                    if result:
                        # Calculate metrics
                        metrics = self._calculate_comprehensive_metrics(
                            scenario['y_measured'], 
                            result['predictions'],
                            execution_time
                        )
                        
                        method_results.append({
                            'scenario': scenario,
                            'result': result,
                            'metrics': metrics,
                            'execution_time': execution_time,
                            'success': True
                        })
                        print(f"    ✅ Success: R²={metrics['r2']:.4f}, "
                              f"Time={execution_time:.2f}s")
                    else:
                        method_results.append({
                            'scenario': scenario,
                            'result': None,
                            'metrics': None,
                            'execution_time': execution_time,
                            'success': False
                        })
                        print(f"    ❌ Failed")
                        
                except Exception as e:
                    method_results.append({
                        'scenario': scenario,
                        'result': None,
                        'metrics': None,
                        'execution_time': 0,
                        'success': False,
                        'error': str(e)
                    })
                    print(f"    ❌ Error: {str(e)}")
            
            all_results[method_name] = method_results
        
        self.results = all_results
        return all_results
    
    def _calculate_comprehensive_metrics(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, 
                                       execution_time: float) -> Dict[str, float]:
        """Calculate comprehensive metrics for evaluation."""
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
        
        # Frequency domain metrics
        fft_true = jnp.fft.fft(y_true, axis=0)
        fft_pred = jnp.fft.fft(y_pred, axis=0)
        freq_error = jnp.mean(jnp.abs(fft_true - fft_pred))
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'nrmse': float(nrmse),
            'correlation': float(correlation),
            'freq_error': float(freq_error),
            'execution_time': execution_time
        }
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results with statistical evaluation."""
        print("\n📈 STATISTICAL ANALYSIS")
        print("=" * 50)
        
        analysis = {}
        
        for method_name, method_results in self.results.items():
            successful_results = [r for r in method_results if r['success']]
            
            if not successful_results:
                print(f"{method_name}: No successful runs")
                continue
            
            # Extract metrics
            metrics_data = {}
            for metric_name in ['r2', 'mse', 'mae', 'nrmse', 'correlation', 'execution_time']:
                values = [r['metrics'][metric_name] for r in successful_results]
                metrics_data[metric_name] = values
            
            # Statistical analysis
            stats_summary = {}
            for metric_name, values in metrics_data.items():
                stats_summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
            
            analysis[method_name] = {
                'success_rate': len(successful_results) / len(method_results),
                'n_successful': len(successful_results),
                'n_total': len(method_results),
                'metrics': metrics_data,
                'statistics': stats_summary
            }
            
            print(f"\n{method_name}:")
            print(f"  Success Rate: {analysis[method_name]['success_rate']:.2%}")
            print(f"  R²: {stats_summary['r2']['mean']:.4f} ± {stats_summary['r2']['std']:.4f}")
            print(f"  MSE: {stats_summary['mse']['mean']:.6f} ± {stats_summary['mse']['std']:.6f}")
            print(f"  Time: {stats_summary['execution_time']['mean']:.2f}s ± {stats_summary['execution_time']['std']:.2f}s")
        
        return analysis
    
    def generate_comparison_report(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive comparison report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE SYSTEM IDENTIFICATION BENCHMARK REPORT")
        report.append("=" * 80)
        
        # Method ranking
        report.append("\n🏆 METHOD RANKING")
        report.append("-" * 40)
        
        # Rank by mean R²
        method_scores = []
        for method_name, method_analysis in analysis.items():
            if method_analysis['success_rate'] > 0:
                mean_r2 = method_analysis['statistics']['r2']['mean']
                method_scores.append((method_name, mean_r2, method_analysis['success_rate']))
        
        method_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (method_name, r2, success_rate) in enumerate(method_scores, 1):
            report.append(f"{i}. {method_name:20s} | R² = {r2:8.4f} | Success = {success_rate:6.2%}")
        
        # Detailed statistics
        report.append("\n📊 DETAILED STATISTICS")
        report.append("-" * 40)
        
        for method_name, method_analysis in analysis.items():
            if method_analysis['success_rate'] > 0:
                report.append(f"\n{method_name}:")
                stats = method_analysis['statistics']
                
                report.append(f"  R²:           {stats['r2']['mean']:8.4f} ± {stats['r2']['std']:8.4f}")
                report.append(f"  MSE:          {stats['mse']['mean']:8.6f} ± {stats['mse']['std']:8.6f}")
                report.append(f"  MAE:          {stats['mae']['mean']:8.6f} ± {stats['mae']['std']:8.6f}")
                report.append(f"  NRMSE:        {stats['nrmse']['mean']:8.4f} ± {stats['nrmse']['std']:8.4f}")
                report.append(f"  Correlation:  {stats['correlation']['mean']:8.4f} ± {stats['correlation']['std']:8.4f}")
                report.append(f"  Time (s):     {stats['execution_time']['mean']:8.2f} ± {stats['execution_time']['std']:8.2f}")
                report.append(f"  Success Rate: {method_analysis['success_rate']:8.2%}")
        
        # Robustness analysis
        report.append("\n🛡️ ROBUSTNESS ANALYSIS")
        report.append("-" * 40)
        
        for method_name, method_analysis in analysis.items():
            if method_analysis['success_rate'] > 0:
                r2_std = method_analysis['statistics']['r2']['std']
                robustness = "High" if r2_std < 0.05 else "Medium" if r2_std < 0.1 else "Low"
                report.append(f"{method_name:20s}: {robustness:6s} robustness (σ_R² = {r2_std:.4f})")
        
        # Performance summary
        report.append("\n⚡ PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        fastest_method = min(analysis.items(), 
                           key=lambda x: x[1]['statistics']['execution_time']['mean'] 
                           if x[1]['success_rate'] > 0 else float('inf'))
        most_accurate = max(analysis.items(), 
                          key=lambda x: x[1]['statistics']['r2']['mean'] 
                          if x[1]['success_rate'] > 0 else -1)
        most_robust = min(analysis.items(), 
                        key=lambda x: x[1]['statistics']['r2']['std'] 
                        if x[1]['success_rate'] > 0 else float('inf'))
        
        report.append(f"Fastest Method:     {fastest_method[0]}")
        report.append(f"Most Accurate:      {most_accurate[0]}")
        report.append(f"Most Robust:        {most_robust[0]}")
        
        return "\n".join(report)


def run_comprehensive_benchmark():
    """Run comprehensive benchmark of all methods."""
    print("🚀 PHASE 4: COMPREHENSIVE BENCHMARKING")
    print("=" * 80)
    
    # Initialize benchmark
    benchmark = ComprehensiveBenchmark()
    
    # Register methods
    benchmark.register_method(
        "Baseline_Nonlinear", 
        lambda u, y: {'predictions': NonlinearLoudspeakerModel().predict(u)}
    )
    
    benchmark.register_method(
        "Fast_Bayesian", 
        fast_bayesian_identification_method,
        num_samples=20  # Reduced for faster benchmarking
    )
    
    benchmark.register_method(
        "GP_Surrogate", 
        gp_surrogate_identification_method
    )
    
    # Generate test scenarios
    print("\n📊 Generating test scenarios...")
    scenarios = benchmark.generate_test_scenarios(n_scenarios=8)  # Reduced for faster execution
    print(f"Generated {len(scenarios)} diverse test scenarios")
    
    # Run benchmark
    results = benchmark.run_benchmark(scenarios)
    
    # Analyze results
    analysis = benchmark.analyze_results()
    
    # Generate report
    report = benchmark.generate_comparison_report(analysis)
    
    print("\n" + report)
    
    return {
        'benchmark': benchmark,
        'results': results,
        'analysis': analysis,
        'report': report
    }


if __name__ == "__main__":
    # Run comprehensive benchmark
    benchmark_results = run_comprehensive_benchmark()
    
    print("\n🎉 PHASE 4 BENCHMARKING COMPLETED!")
    print("=" * 80)
    print("Comprehensive evaluation of all system identification methods")
    print("Statistical analysis and performance comparison completed")
    print("Ready for final documentation and publication!")
    print("=" * 80)
