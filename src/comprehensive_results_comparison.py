"""
Comprehensive Results Comparison with Spectrograms and Complex Tone Analysis.

This module provides a complete comparison of all system identification methods
with spectrograms, complex tone signals (15Hz + 600Hz), and detailed R² analysis.

Author: Research Team
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
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


class ComprehensiveResultsAnalyzer:
    """
    Comprehensive analysis of all system identification methods with spectrograms.
    """
    
    def __init__(self, results_dir: str = "results"):
        """Initialize results analyzer."""
        self.results_dir = results_dir
        self.ground_truth = create_standard_ground_truth()
        self.results = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/spectrograms", exist_ok=True)
        os.makedirs(f"{results_dir}/comparisons", exist_ok=True)
    
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
                'Re': baseline_model.Re,
                'Le': baseline_model.Le,
                'Bl': baseline_model.Bl,
                'M': baseline_model.M,
                'K': baseline_model.K,
                'Rm': baseline_model.Rm,
                'L20': baseline_model.L20,
                'R20': baseline_model.R20
            },
            'execution_time': baseline_time,
            'method_name': 'Baseline Nonlinear'
        }
        
        # 2. Fast Bayesian Inference
        print("  2. Fast Bayesian Inference...")
        start_time = time.time()
        bayesian_result = fast_bayesian_identification_method(u, y, num_samples=20)
        bayesian_time = time.time() - start_time
        
        if bayesian_result:
            results['bayesian'] = {
                'predictions': bayesian_result['predictions'],
                'parameters': bayesian_result['parameters'],
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
    
    def create_spectrograms(self, test_data: Dict[str, jnp.ndarray], 
                          results: Dict[str, Any], signal_name: str):
        """Create spectrograms for input, output, and predictions."""
        print(f"📊 Creating spectrograms for {signal_name}...")
        
        u = test_data['u']
        y_true = test_data['y_true']
        y_measured = test_data['y_measured']
        t = test_data['t']
        dt = t[1] - t[0]
        fs = 1 / dt
        
        # Create figure with subplots
        n_methods = len(results) + 2  # +2 for input and ground truth
        fig, axes = plt.subplots(n_methods, 2, figsize=(15, 3 * n_methods))
        
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        # Plot input signal spectrogram
        f_input, t_spec, Sxx_input = signal.spectrogram(u, fs, nperseg=256, noverlap=128)
        im1 = axes[0, 0].pcolormesh(t_spec, f_input, 10 * np.log10(Sxx_input), shading='gouraud')
        axes[0, 0].set_title(f'Input Signal Spectrogram ({signal_name})')
        axes[0, 0].set_ylabel('Frequency [Hz]')
        axes[0, 0].set_xlabel('Time [s]')
        plt.colorbar(im1, ax=axes[0, 0], label='Power [dB]')
        
        # Plot input time series
        axes[0, 1].plot(t, u)
        axes[0, 1].set_title(f'Input Signal Time Series ({signal_name})')
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].set_ylabel('Amplitude [V]')
        axes[0, 1].grid(True)
        
        # Plot ground truth spectrogram
        f_gt, t_spec, Sxx_gt = signal.spectrogram(y_true[:, 0], fs, nperseg=256, noverlap=128)
        im2 = axes[1, 0].pcolormesh(t_spec, f_gt, 10 * np.log10(Sxx_gt), shading='gouraud')
        axes[1, 0].set_title('Ground Truth Current Spectrogram')
        axes[1, 0].set_ylabel('Frequency [Hz]')
        axes[1, 0].set_xlabel('Time [s]')
        plt.colorbar(im2, ax=axes[1, 0], label='Power [dB]')
        
        # Plot ground truth time series
        axes[1, 1].plot(t, y_true[:, 0], label='Current', alpha=0.8)
        axes[1, 1].plot(t, y_true[:, 1], label='Velocity', alpha=0.8)
        axes[1, 1].set_title('Ground Truth Time Series')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot method results
        row_idx = 2
        for method_key, method_result in results.items():
            y_pred = method_result['predictions']
            
            # Spectrogram
            f_pred, t_spec, Sxx_pred = signal.spectrogram(y_pred[:, 0], fs, nperseg=256, noverlap=128)
            im = axes[row_idx, 0].pcolormesh(t_spec, f_pred, 10 * np.log10(Sxx_pred), shading='gouraud')
            axes[row_idx, 0].set_title(f'{method_result["method_name"]} Current Spectrogram')
            axes[row_idx, 0].set_ylabel('Frequency [Hz]')
            axes[row_idx, 0].set_xlabel('Time [s]')
            plt.colorbar(im, ax=axes[row_idx, 0], label='Power [dB]')
            
            # Time series comparison
            axes[row_idx, 1].plot(t, y_true[:, 0], label='Ground Truth', alpha=0.7)
            axes[row_idx, 1].plot(t, y_pred[:, 0], label='Predicted', alpha=0.7)
            axes[row_idx, 1].set_title(f'{method_result["method_name"]} Comparison')
            axes[row_idx, 1].set_xlabel('Time [s]')
            axes[row_idx, 1].set_ylabel('Current [A]')
            axes[row_idx, 1].legend()
            axes[row_idx, 1].grid(True)
            
            row_idx += 1
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/spectrograms/{signal_name}_spectrograms.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Spectrograms saved to {self.results_dir}/spectrograms/{signal_name}_spectrograms.png")
    
    def create_r2_comparison(self, all_results: Dict[str, Dict[str, Any]]):
        """Create R² comparison across all methods and signals."""
        print("📊 Creating R² comparison...")
        
        # Prepare data for comparison
        methods = []
        signals = []
        r2_values = []
        execution_times = []
        
        for signal_name, signal_results in all_results.items():
            for method_key, method_result in signal_results.items():
                if 'predictions' in method_result:
                    # Calculate metrics
                    test_data = self.get_test_data(signal_name)
                    metrics = self.calculate_comprehensive_metrics(
                        test_data['y_measured'], method_result['predictions']
                    )
                    
                    methods.append(method_result['method_name'])
                    signals.append(signal_name.replace('_', ' ').title())
                    r2_values.append(metrics['r2'])
                    execution_times.append(method_result['execution_time'])
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² comparison
        unique_methods = list(set(methods))
        unique_signals = list(set(signals))
        
        # Create grouped bar chart
        x = np.arange(len(unique_signals))
        width = 0.25
        
        for i, method in enumerate(unique_methods):
            method_r2 = []
            for signal in unique_signals:
                # Find R² for this method-signal combination
                for j, (m, s) in enumerate(zip(methods, signals)):
                    if m == method and s == signal:
                        method_r2.append(r2_values[j])
                        break
                else:
                    method_r2.append(0.0)  # Default if not found
            
            ax1.bar(x + i * width, method_r2, width, label=method, alpha=0.8)
        
        ax1.set_xlabel('Signal Type')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Comparison Across Methods and Signals')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(unique_signals)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Execution time comparison
        for i, method in enumerate(unique_methods):
            method_times = []
            for signal in unique_signals:
                # Find execution time for this method-signal combination
                for j, (m, s) in enumerate(zip(methods, signals)):
                    if m == method and s == signal:
                        method_times.append(execution_times[j])
                        break
                else:
                    method_times.append(0.0)  # Default if not found
            
            ax2.bar(x + i * width, method_times, width, label=method, alpha=0.8)
        
        ax2.set_xlabel('Signal Type')
        ax2.set_ylabel('Execution Time [s]')
        ax2.set_title('Execution Time Comparison')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(unique_signals)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/comparisons/r2_and_time_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ R² comparison saved to {self.results_dir}/comparisons/r2_and_time_comparison.png")
    
    def create_error_timeseries_comparison(self, all_results: Dict[str, Dict[str, Any]]):
        """Create comprehensive error timeseries comparison."""
        print("📊 Creating error timeseries comparison...")
        
        n_signals = len(all_results)
        n_methods = max(len(results) for results in all_results.values())
        
        fig, axes = plt.subplots(n_signals, n_methods, figsize=(5 * n_methods, 4 * n_signals))
        
        if n_signals == 1:
            axes = axes.reshape(1, -1)
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
        
        for signal_idx, (signal_name, signal_results) in enumerate(all_results.items()):
            test_data = self.get_test_data(signal_name)
            y_true = test_data['y_measured']
            t = test_data['t']
            
            for method_idx, (method_key, method_result) in enumerate(signal_results.items()):
                if 'predictions' in method_result:
                    y_pred = method_result['predictions']
                    error = y_true - y_pred
                    
                    # Calculate metrics
                    metrics = self.calculate_comprehensive_metrics(y_true, y_pred)
                    
                    # Plot error timeseries
                    axes[signal_idx, method_idx].plot(t, error[:, 0], label='Current Error', alpha=0.7)
                    axes[signal_idx, method_idx].plot(t, error[:, 1], label='Velocity Error', alpha=0.7)
                    axes[signal_idx, method_idx].set_title(f'{method_result["method_name"]}\n{signal_name.replace("_", " ").title()}\nR² = {metrics["r2"]:.4f}')
                    axes[signal_idx, method_idx].set_xlabel('Time [s]')
                    axes[signal_idx, method_idx].set_ylabel('Error')
                    axes[signal_idx, method_idx].legend()
                    axes[signal_idx, method_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/comparisons/error_timeseries_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Error timeseries comparison saved to {self.results_dir}/comparisons/error_timeseries_comparison.png")
    
    def save_comprehensive_results(self, all_results: Dict[str, Dict[str, Any]]):
        """Save comprehensive results to JSON."""
        print("💾 Saving comprehensive results...")
        
        # Prepare results for JSON serialization
        json_results = {}
        
        for signal_name, signal_results in all_results.items():
            json_results[signal_name] = {}
            
            for method_key, method_result in signal_results.items():
                if 'predictions' in method_result:
                    # Calculate metrics
                    test_data = self.get_test_data(signal_name)
                    metrics = self.calculate_comprehensive_metrics(
                        test_data['y_measured'], method_result['predictions']
                    )
                    
                    json_results[signal_name][method_key] = {
                        'method_name': method_result['method_name'],
                        'metrics': metrics,
                        'execution_time': method_result['execution_time'],
                        'parameters': {k: float(v) if isinstance(v, (int, float, jnp.ndarray)) else str(v) 
                                     for k, v in method_result['parameters'].items()},
                        'error_timeseries': {
                            'current': method_result['predictions'][:, 0].tolist(),
                            'velocity': method_result['predictions'][:, 1].tolist()
                        }
                    }
        
        # Save to JSON
        with open(f'{self.results_dir}/comprehensive_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"  ✅ Results saved to {self.results_dir}/comprehensive_results.json")
    
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
        
        # Run all methods on both signals
        all_results = {}
        
        print("\n🔬 Running methods on complex tone signal...")
        all_results['complex_tone'] = self.run_all_methods(complex_tone_data)
        
        print("\n🔬 Running methods on pink noise signal...")
        all_results['pink_noise'] = self.run_all_methods(pink_noise_data)
        
        # Create spectrograms
        print("\n📊 Creating spectrograms...")
        self.create_spectrograms(complex_tone_data, all_results['complex_tone'], 'complex_tone')
        self.create_spectrograms(pink_noise_data, all_results['pink_noise'], 'pink_noise')
        
        # Create comparisons
        print("\n📊 Creating comparisons...")
        self.create_r2_comparison(all_results)
        self.create_error_timeseries_comparison(all_results)
        
        # Save results
        self.save_comprehensive_results(all_results)
        
        # Print summary
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 80)
        
        for signal_name, signal_results in all_results.items():
            print(f"\n{signal_name.replace('_', ' ').title()} Results:")
            print("-" * 40)
            
            for method_key, method_result in signal_results.items():
                if 'predictions' in method_result:
                    test_data = self.get_test_data(signal_name)
                    metrics = self.calculate_comprehensive_metrics(
                        test_data['y_measured'], method_result['predictions']
                    )
                    
                    print(f"  {method_result['method_name']}:")
                    print(f"    R² = {metrics['r2']:.4f}")
                    print(f"    MSE = {metrics['mse']:.6f}")
                    print(f"    Time = {method_result['execution_time']:.2f}s")
        
        print(f"\n✅ All results saved to {self.results_dir}/")
        print("✅ Spectrograms created for both signals")
        print("✅ R² comparison across all methods")
        print("✅ Error timeseries analysis completed")
        
        return all_results


def main():
    """Run comprehensive results analysis."""
    # Create analyzer
    analyzer = ComprehensiveResultsAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
    print("=" * 80)
    print("All methods compared on complex tone (15Hz + 600Hz) and pink noise")
    print("Spectrograms created for frequency domain analysis")
    print("R² comparison across all methods and signals")
    print("Error timeseries analysis for all combinations")
    print("Results saved to 'results/' directory")
    print("=" * 80)


if __name__ == "__main__":
    main()
