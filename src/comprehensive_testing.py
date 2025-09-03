"""
Comprehensive Testing Framework for Loudspeaker System Identification.

This module provides comprehensive testing and evaluation of different system
identification methods with specific metrics as requested:
- Model parameters
- Error timeseries
- Final loss
- Final R²

Author: Research Team
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.ground_truth_model import GroundTruthLoudspeakerModel


@dataclass
class TestResult:
    """Container for test results with required metrics."""
    
    # Method identification
    method_name: str
    framework: str
    
    # Model parameters (fitted)
    model_parameters: Dict[str, Any]
    
    # Error timeseries
    error_timeseries: Dict[str, jnp.ndarray]  # 'current', 'velocity'
    
    # Final metrics
    final_loss: float
    final_r2: float
    
    # Additional metrics
    nrmse: Dict[str, float]  # Normalized RMSE for each output
    mae: Dict[str, float]    # Mean Absolute Error
    correlation: Dict[str, float]  # Correlation coefficient
    
    # Performance metrics
    training_time: float
    convergence_info: Dict[str, Any]
    
    # Data info
    n_samples: int
    sample_rate: float
    excitation_type: str


@dataclass
class ComparisonResult:
    """Container for method comparison results."""
    
    results: List[TestResult]
    ground_truth: GroundTruthLoudspeakerModel
    test_data: Dict[str, jnp.ndarray]
    
    # Summary statistics
    method_ranking: List[str]
    best_method: str
    performance_summary: Dict[str, Dict[str, float]]


class ComprehensiveTester:
    """
    Comprehensive testing framework for loudspeaker system identification methods.
    
    This class provides methods to test different system identification approaches
    and compare their performance using standardized metrics.
    """
    
    def __init__(self, ground_truth_model: GroundTruthLoudspeakerModel):
        """Initialize tester with ground truth model."""
        self.ground_truth = ground_truth_model
        self.results: List[TestResult] = []
    
    def generate_test_data(self, excitation_type: str = 'pink_noise', 
                          duration: float = 1.0, sample_rate: float = 48000,
                          amplitude: float = 2.0, noise_level: float = 0.01) -> Dict[str, jnp.ndarray]:
        """
        Generate test data using ground truth model.
        
        Args:
            excitation_type: Type of excitation ('pink_noise', 'sine', 'chirp', 'multitone')
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Excitation amplitude
            noise_level: Relative noise level
            
        Returns:
            Dictionary with test data
        """
        n_samples = int(duration * sample_rate)
        dt = 1.0 / sample_rate
        t = jnp.linspace(0, duration, n_samples)
        
        # Generate excitation signal
        if excitation_type == 'pink_noise':
            u = self._generate_pink_noise(n_samples, amplitude)
        elif excitation_type == 'sine':
            u = amplitude * jnp.sin(2 * jnp.pi * 100 * t)  # 100 Hz sine
        elif excitation_type == 'chirp':
            f0, f1 = 10, 1000
            u = amplitude * jnp.sin(2 * jnp.pi * (f0 + (f1 - f0) * t / duration) * t)
        elif excitation_type == 'multitone':
            u = self._generate_multitone(n_samples, amplitude)
        else:
            raise ValueError(f"Unknown excitation type: {excitation_type}")
        
        # Generate synthetic data
        synthetic_data = self.ground_truth.generate_synthetic_data(
            u, x0=jnp.zeros(4), dt=dt, noise_level=noise_level
        )
        
        return {
            'time': synthetic_data['time'],
            'voltage': synthetic_data['voltage'],
            'current_measured': synthetic_data['current_measured'],
            'velocity_measured': synthetic_data['velocity_measured'],
            'current_clean': synthetic_data['current_clean'],
            'velocity_clean': synthetic_data['velocity_clean'],
            'excitation_type': excitation_type,
            'sample_rate': sample_rate,
            'amplitude': amplitude,
            'noise_level': noise_level
        }
    
    def _generate_pink_noise(self, n_samples: int, amplitude: float) -> jnp.ndarray:
        """Generate pink noise excitation."""
        rng_key = jax.random.PRNGKey(42)
        white_noise = jax.random.normal(rng_key, (n_samples,))
        
        # Simple pink noise filter (1/f)
        freqs = jnp.fft.fftfreq(n_samples)
        pink_filter = 1 / jnp.sqrt(jnp.abs(freqs) + 1e-10)
        pink_filter = pink_filter.at[0].set(0)  # Remove DC
        
        pink_noise = jnp.real(jnp.fft.ifft(jnp.fft.fft(white_noise) * pink_filter))
        return amplitude * pink_noise / jnp.std(pink_noise)
    
    def _generate_multitone(self, n_samples: int, amplitude: float) -> jnp.ndarray:
        """Generate multitone excitation."""
        t = jnp.linspace(0, 1, n_samples)
        frequencies = jnp.array([50, 100, 200, 500, 1000, 2000])  # Hz
        
        signal = jnp.zeros(n_samples)
        for f in frequencies:
            signal += jnp.sin(2 * jnp.pi * f * t)
        
        return amplitude * signal / jnp.max(jnp.abs(signal))
    
    def test_method(self, method_func: Callable, method_name: str, framework: str,
                   test_data: Dict[str, jnp.ndarray], **kwargs) -> TestResult:
        """
        Test a specific system identification method.
        
        Args:
            method_func: Function that implements the identification method
            method_name: Name of the method
            framework: Framework used (e.g., 'Diffrax+JAXopt', 'Dynax', 'NumPyro')
            test_data: Test data dictionary
            **kwargs: Additional arguments for the method
            
        Returns:
            TestResult with all required metrics
        """
        print(f"Testing {method_name} ({framework})...")
        
        # Prepare data
        u = test_data['voltage']
        y_measured = jnp.stack([
            test_data['current_measured'],
            test_data['velocity_measured']
        ], axis=1)
        y_clean = jnp.stack([
            test_data['current_clean'],
            test_data['velocity_clean']
        ], axis=1)
        
        # Run identification method
        start_time = time.time()
        try:
            result = method_func(u, y_measured, **kwargs)
            training_time = time.time() - start_time
            
            # Extract fitted model and predictions
            if isinstance(result, dict):
                fitted_model = result.get('model')
                y_pred = result.get('predictions')
                model_params = result.get('parameters', {})
                convergence_info = result.get('convergence', {})
            else:
                # Assume result is the fitted model
                fitted_model = result
                y_pred = self._predict_with_model(fitted_model, u, test_data)
                model_params = self._extract_parameters(fitted_model)
                convergence_info = {}
            
            # Calculate errors
            error_timeseries = {
                'current': y_clean[:, 0] - y_pred[:, 0],
                'velocity': y_clean[:, 1] - y_pred[:, 1]
            }
            
            # Calculate metrics
            final_loss = self._calculate_loss(y_clean, y_pred)
            final_r2 = self._calculate_r2(y_clean, y_pred)
            nrmse = self._calculate_nrmse(y_clean, y_pred)
            mae = self._calculate_mae(y_clean, y_pred)
            correlation = self._calculate_correlation(y_clean, y_pred)
            
            test_result = TestResult(
                method_name=method_name,
                framework=framework,
                model_parameters=model_params,
                error_timeseries=error_timeseries,
                final_loss=final_loss,
                final_r2=final_r2,
                nrmse=nrmse,
                mae=mae,
                correlation=correlation,
                training_time=training_time,
                convergence_info=convergence_info,
                n_samples=len(u),
                sample_rate=test_data['sample_rate'],
                excitation_type=test_data['excitation_type']
            )
            
            self.results.append(test_result)
            print(f"  ✓ Completed in {training_time:.2f}s, R² = {final_r2:.4f}")
            return test_result
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            # Return failed result
            return TestResult(
                method_name=method_name,
                framework=framework,
                model_parameters={},
                error_timeseries={'current': jnp.array([]), 'velocity': jnp.array([])},
                final_loss=float('inf'),
                final_r2=-float('inf'),
                nrmse={'current': float('inf'), 'velocity': float('inf')},
                mae={'current': float('inf'), 'velocity': float('inf')},
                correlation={'current': 0.0, 'velocity': 0.0},
                training_time=time.time() - start_time,
                convergence_info={'error': str(e)},
                n_samples=len(u),
                sample_rate=test_data['sample_rate'],
                excitation_type=test_data['excitation_type']
            )
    
    def _predict_with_model(self, model: Any, u: jnp.ndarray, test_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Predict using fitted model."""
        # This is a placeholder - will be implemented for each specific model type
        # For now, return zeros
        return jnp.zeros((len(u), 2))
    
    def _extract_parameters(self, model: Any) -> Dict[str, Any]:
        """Extract parameters from fitted model."""
        # This is a placeholder - will be implemented for each specific model type
        return {}
    
    def _calculate_loss(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """Calculate mean squared error loss."""
        return float(jnp.mean((y_true - y_pred) ** 2))
    
    def _calculate_r2(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """Calculate R² score."""
        # Calculate R² for each output separately, then average
        r2_scores = []
        for i in range(y_true.shape[1]):
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            r2_scores.append(r2)
        return float(jnp.mean(jnp.array(r2_scores)))
    
    def _calculate_nrmse(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> Dict[str, float]:
        """Calculate normalized RMSE for each output."""
        nrmse = {}
        output_names = ['current', 'velocity']
        
        for i, name in enumerate(output_names):
            mse = jnp.mean((y_true[:, i] - y_pred[:, i]) ** 2)
            rmse = jnp.sqrt(mse)
            std_true = jnp.std(y_true[:, i])
            nrmse[name] = float(rmse / std_true)
        
        return nrmse
    
    def _calculate_mae(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> Dict[str, float]:
        """Calculate mean absolute error for each output."""
        mae = {}
        output_names = ['current', 'velocity']
        
        for i, name in enumerate(output_names):
            mae[name] = float(jnp.mean(jnp.abs(y_true[:, i] - y_pred[:, i])))
        
        return mae
    
    def _calculate_correlation(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> Dict[str, float]:
        """Calculate correlation coefficient for each output."""
        correlation = {}
        output_names = ['current', 'velocity']
        
        for i, name in enumerate(output_names):
            corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
            correlation[name] = float(corr)
        
        return correlation
    
    def compare_methods(self, test_data: Dict[str, jnp.ndarray]) -> ComparisonResult:
        """
        Compare all tested methods and generate ranking.
        
        Args:
            test_data: Test data used for comparison
            
        Returns:
            ComparisonResult with method comparison
        """
        if not self.results:
            raise ValueError("No test results available. Run test_method() first.")
        
        # Rank methods by R² score
        method_ranking = sorted(
            self.results,
            key=lambda x: x.final_r2,
            reverse=True
        )
        
        best_method = method_ranking[0].method_name if method_ranking else "None"
        
        # Generate performance summary
        performance_summary = {}
        for result in self.results:
            performance_summary[result.method_name] = {
                'r2': result.final_r2,
                'loss': result.final_loss,
                'nrmse_current': result.nrmse['current'],
                'nrmse_velocity': result.nrmse['velocity'],
                'training_time': result.training_time
            }
        
        return ComparisonResult(
            results=self.results,
            ground_truth=self.ground_truth,
            test_data=test_data,
            method_ranking=[r.method_name for r in method_ranking],
            best_method=best_method,
            performance_summary=performance_summary
        )
    
    def generate_report(self, comparison_result: ComparisonResult, 
                       output_dir: str = "test_results") -> str:
        """
        Generate comprehensive test report.
        
        Args:
            comparison_result: Results from compare_methods()
            output_dir: Directory to save report
            
        Returns:
            Path to generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate plots
        self._plot_error_timeseries(comparison_result, output_path)
        self._plot_parameter_comparison(comparison_result, output_path)
        self._plot_performance_comparison(comparison_result, output_path)
        
        # Generate JSON report
        report_data = {
            'ground_truth_parameters': self.ground_truth.get_parameters(),
            'test_data_info': {
                'n_samples': comparison_result.test_data['time'].shape[0],
                'sample_rate': comparison_result.test_data['sample_rate'],
                'excitation_type': comparison_result.test_data['excitation_type'],
                'amplitude': comparison_result.test_data['amplitude'],
                'noise_level': comparison_result.test_data['noise_level']
            },
            'method_ranking': comparison_result.method_ranking,
            'best_method': comparison_result.best_method,
            'performance_summary': comparison_result.performance_summary,
            'detailed_results': []
        }
        
        # Add detailed results
        for result in comparison_result.results:
            result_dict = {
                'method_name': result.method_name,
                'framework': result.framework,
                'model_parameters': {k: float(v) if isinstance(v, (int, float, jnp.ndarray)) else v 
                                   for k, v in result.model_parameters.items()},
                'final_loss': result.final_loss,
                'final_r2': result.final_r2,
                'nrmse': result.nrmse,
                'mae': result.mae,
                'correlation': result.correlation,
                'training_time': result.training_time,
                'convergence_info': result.convergence_info
            }
            report_data['detailed_results'].append(result_dict)
        
        # Save JSON report
        report_file = output_path / "comprehensive_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate text summary
        summary_file = output_path / "test_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE LOUDSPEAKER SYSTEM IDENTIFICATION TEST RESULTS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Ground Truth Model Parameters:\n")
            for param, value in self.ground_truth.get_parameters().items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            
            f.write(f"Test Data:\n")
            f.write(f"  Samples: {comparison_result.test_data['time'].shape[0]}\n")
            f.write(f"  Sample Rate: {comparison_result.test_data['sample_rate']} Hz\n")
            f.write(f"  Excitation: {comparison_result.test_data['excitation_type']}\n")
            f.write(f"  Amplitude: {comparison_result.test_data['amplitude']} V\n")
            f.write(f"  Noise Level: {comparison_result.test_data['noise_level']}\n\n")
            
            f.write("Method Ranking (by R² score):\n")
            for i, method in enumerate(comparison_result.method_ranking, 1):
                result = next(r for r in comparison_result.results if r.method_name == method)
                f.write(f"  {i}. {method} ({result.framework}): R² = {result.final_r2:.4f}\n")
            f.write("\n")
            
            f.write("Detailed Results:\n")
            for result in comparison_result.results:
                f.write(f"\n{result.method_name} ({result.framework}):\n")
                f.write(f"  R² Score: {result.final_r2:.4f}\n")
                f.write(f"  Final Loss: {result.final_loss:.6f}\n")
                f.write(f"  NRMSE - Current: {result.nrmse['current']:.4f}\n")
                f.write(f"  NRMSE - Velocity: {result.nrmse['velocity']:.4f}\n")
                f.write(f"  Training Time: {result.training_time:.2f}s\n")
                f.write(f"  Model Parameters: {result.model_parameters}\n")
        
        print(f"Comprehensive test report generated in: {output_path}")
        return str(output_path)
    
    def _plot_error_timeseries(self, comparison_result: ComparisonResult, output_path: Path):
        """Plot error timeseries for all methods."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        for result in comparison_result.results:
            if len(result.error_timeseries['current']) > 0:
                time = comparison_result.test_data['time']
                axes[0].plot(time, result.error_timeseries['current'], 
                           label=f"{result.method_name} ({result.framework})", alpha=0.7)
                axes[1].plot(time, result.error_timeseries['velocity'], 
                           label=f"{result.method_name} ({result.framework})", alpha=0.7)
        
        axes[0].set_title('Current Error Timeseries')
        axes[0].set_ylabel('Error [A]')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_title('Velocity Error Timeseries')
        axes[1].set_xlabel('Time [s]')
        axes[1].set_ylabel('Error [m/s]')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / "error_timeseries.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_comparison(self, comparison_result: ComparisonResult, output_path: Path):
        """Plot parameter comparison with ground truth."""
        gt_params = self.ground_truth.get_parameters()
        linear_params = ['Re', 'Le', 'Bl', 'M', 'K', 'Rm']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, param in enumerate(linear_params):
            gt_value = gt_params[param]
            methods = []
            values = []
            
            for result in comparison_result.results:
                if param in result.model_parameters:
                    methods.append(f"{result.method_name}\n({result.framework})")
                    values.append(result.model_parameters[param])
            
            if values:
                axes[i].bar(methods, values, alpha=0.7)
                axes[i].axhline(y=gt_value, color='red', linestyle='--', 
                              label=f'Ground Truth: {gt_value:.4f}')
                axes[i].set_title(f'{param} Parameter Comparison')
                axes[i].set_ylabel(f'{param} [{self._get_parameter_unit(param)}]')
                axes[i].legend()
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / "parameter_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, comparison_result: ComparisonResult, output_path: Path):
        """Plot performance comparison."""
        methods = [r.method_name for r in comparison_result.results]
        r2_scores = [r.final_r2 for r in comparison_result.results]
        training_times = [r.training_time for r in comparison_result.results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² scores
        bars1 = ax1.bar(methods, r2_scores, alpha=0.7)
        ax1.set_title('R² Score Comparison')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Training times
        bars2 = ax2.bar(methods, training_times, alpha=0.7, color='orange')
        ax2.set_title('Training Time Comparison')
        ax2.set_ylabel('Training Time [s]')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars2, training_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.01,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_parameter_unit(self, param: str) -> str:
        """Get unit for parameter."""
        units = {
            'Re': 'Ω', 'Le': 'H', 'Bl': 'N/A', 'M': 'kg',
            'K': 'N/m', 'Rm': 'N·s/m', 'L20': 'H', 'R20': 'Ω'
        }
        return units.get(param, '')
