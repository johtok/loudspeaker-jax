"""
Comprehensive System Identification Framework.

This module integrates all system identification methods:
- Nonlinear optimization (Gauss-Newton, Levenberg-Marquardt, L-BFGS)
- Bayesian inference (NumPyro, BlackJAX)
- Gaussian Process surrogates (GPJax)
- State-space modeling (Dynamax)

Author: Research Team
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Callable
from dataclasses import dataclass
import time
from pathlib import Path

# Import all identification methods
from nonlinear_loudspeaker_model import (
    NonlinearLoudspeakerModel, 
    NonlinearSystemIdentifier,
    nonlinear_identification_method
)
from bayesian_inference import (
    BayesianLoudspeakerInference,
    bayesian_identification_method
)
from gp_surrogate_modeling import (
    GPSurrogateModel,
    HybridPhysicsGPModel,
    gp_surrogate_identification_method
)
from state_space_modeling import (
    DynamaxSystemIdentifier,
    dynamax_identification_method
)
from dynax_identification import (
    DynaxSystemIdentifier as OriginalDynaxIdentifier,
    dynax_identification_method as original_dynax_method
)


@dataclass
class IdentificationResult:
    """Container for identification results."""
    method_name: str
    parameters: Dict[str, Any]
    predictions: jnp.ndarray
    error_timeseries: jnp.ndarray
    final_loss: float
    final_r2: float
    training_time: float
    convergence_info: Dict[str, Any]
    model: Any = None


class ComprehensiveSystemIdentifier:
    """
    Comprehensive system identification framework.
    
    Integrates all available methods for loudspeaker system identification.
    """
    
    def __init__(self):
        """Initialize comprehensive identifier."""
        self.results = {}
        self.ground_truth_model = None
    
    def set_ground_truth_model(self, model: Any):
        """Set ground truth model for comparison."""
        self.ground_truth_model = model
    
    def run_nonlinear_optimization_methods(self, u: jnp.ndarray, y: jnp.ndarray,
                                         **kwargs) -> Dict[str, IdentificationResult]:
        """
        Run all nonlinear optimization methods.
        
        Args:
            u: Input voltage [V]
            y: Output measurements [current, velocity]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results for each method
        """
        print("=" * 60)
        print("NONLINEAR OPTIMIZATION METHODS")
        print("=" * 60)
        
        methods = ['gauss_newton', 'levenberg_marquardt', 'lbfgs']
        results = {}
        
        for method in methods:
            print(f"\nRunning {method}...")
            start_time = time.time()
            
            try:
                # Run identification
                result = nonlinear_identification_method(u, y, method=method, **kwargs)
                
                # Calculate metrics
                predictions = result['predictions']
                error_timeseries = y - predictions
                final_loss = jnp.mean(error_timeseries ** 2)
                final_r2 = self._calculate_r2(y, predictions)
                training_time = time.time() - start_time
                
                # Store result
                results[method] = IdentificationResult(
                    method_name=f"Nonlinear {method.replace('_', ' ').title()}",
                    parameters=result['parameters'],
                    predictions=predictions,
                    error_timeseries=error_timeseries,
                    final_loss=float(final_loss),
                    final_r2=float(final_r2),
                    training_time=training_time,
                    convergence_info=result['convergence'],
                    model=result['model']
                )
                
                print(f"  ✓ Completed in {training_time:.2f}s, R² = {final_r2:.4f}")
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                results[method] = None
        
        return results
    
    def run_bayesian_methods(self, u: jnp.ndarray, y: jnp.ndarray,
                           **kwargs) -> Dict[str, IdentificationResult]:
        """
        Run all Bayesian inference methods.
        
        Args:
            u: Input voltage [V]
            y: Output measurements [current, velocity]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results for each method
        """
        print("=" * 60)
        print("BAYESIAN INFERENCE METHODS")
        print("=" * 60)
        
        methods = ['numpyro', 'blackjax']
        results = {}
        
        for method in methods:
            print(f"\nRunning {method}...")
            start_time = time.time()
            
            try:
                # Run identification
                result = bayesian_identification_method(u, y, method=method, **kwargs)
                
                # Calculate metrics
                predictions = result['predictions']
                error_timeseries = y - predictions
                final_loss = jnp.mean(error_timeseries ** 2)
                final_r2 = self._calculate_r2(y, predictions)
                training_time = time.time() - start_time
                
                # Store result
                results[method] = IdentificationResult(
                    method_name=f"Bayesian {method.title()}",
                    parameters=result['parameters'],
                    predictions=predictions,
                    error_timeseries=error_timeseries,
                    final_loss=float(final_loss),
                    final_r2=float(final_r2),
                    training_time=training_time,
                    convergence_info=result['convergence'],
                    model=result['model']
                )
                
                print(f"  ✓ Completed in {training_time:.2f}s, R² = {final_r2:.4f}")
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                results[method] = None
        
        return results
    
    def run_gp_surrogate_methods(self, u: jnp.ndarray, y: jnp.ndarray,
                               **kwargs) -> Dict[str, IdentificationResult]:
        """
        Run GP surrogate modeling methods.
        
        Args:
            u: Input voltage [V]
            y: Output measurements [current, velocity]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results for each method
        """
        print("=" * 60)
        print("GAUSSIAN PROCESS SURROGATE METHODS")
        print("=" * 60)
        
        print(f"\nRunning GP surrogate...")
        start_time = time.time()
        
        try:
            # Run identification
            result = gp_surrogate_identification_method(u, y, **kwargs)
            
            # Calculate metrics
            predictions = result['predictions']['total']
            error_timeseries = y - predictions
            final_loss = jnp.mean(error_timeseries ** 2)
            final_r2 = self._calculate_r2(y, predictions)
            training_time = time.time() - start_time
            
            # Store result
            results = {
                'gp_surrogate': IdentificationResult(
                    method_name="GP Surrogate",
                    parameters=result['training_results']['discrepancy_stats'],
                    predictions=predictions,
                    error_timeseries=error_timeseries,
                    final_loss=float(final_loss),
                    final_r2=float(final_r2),
                    training_time=training_time,
                    convergence_info=result['convergence'],
                    model=result['hybrid_model']
                )
            }
            
            print(f"  ✓ Completed in {training_time:.2f}s, R² = {final_r2:.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            results = {'gp_surrogate': None}
        
        return results
    
    def run_state_space_methods(self, u: jnp.ndarray, y: jnp.ndarray,
                              **kwargs) -> Dict[str, IdentificationResult]:
        """
        Run state-space modeling methods.
        
        Args:
            u: Input voltage [V]
            y: Output measurements [current, velocity]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results for each method
        """
        print("=" * 60)
        print("STATE-SPACE MODELING METHODS")
        print("=" * 60)
        
        methods = ['linear', 'nonlinear']
        results = {}
        
        for method in methods:
            print(f"\nRunning Dynamax {method}...")
            start_time = time.time()
            
            try:
                # Run identification
                result = dynamax_identification_method(u, y, model_type=method, **kwargs)
                
                # Calculate metrics
                predictions = result['predictions']['predicted_emissions']
                error_timeseries = y - predictions
                final_loss = jnp.mean(error_timeseries ** 2)
                final_r2 = self._calculate_r2(y, predictions)
                training_time = time.time() - start_time
                
                # Store result
                results[method] = IdentificationResult(
                    method_name=f"Dynamax {method.title()}",
                    parameters=result['parameters'],
                    predictions=predictions,
                    error_timeseries=error_timeseries,
                    final_loss=float(final_loss),
                    final_r2=float(final_r2),
                    training_time=training_time,
                    convergence_info=result['convergence'],
                    model=result
                )
                
                print(f"  ✓ Completed in {training_time:.2f}s, R² = {final_r2:.4f}")
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                results[method] = None
        
        return results
    
    def run_original_dynax_methods(self, u: jnp.ndarray, y: jnp.ndarray,
                                 **kwargs) -> Dict[str, IdentificationResult]:
        """
        Run original Dynax methods from the reference implementation.
        
        Args:
            u: Input voltage [V]
            y: Output measurements [current, velocity]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results for each method
        """
        print("=" * 60)
        print("ORIGINAL DYNAX METHODS")
        print("=" * 60)
        
        methods = ['dynax_full', 'dynax_linear_only']
        results = {}
        
        for method in methods:
            print(f"\nRunning {method}...")
            start_time = time.time()
            
            try:
                # Run identification
                result = original_dynax_method(u, y, method=method, **kwargs)
                
                # Calculate metrics
                predictions = result['predictions']
                error_timeseries = y - predictions
                final_loss = jnp.mean(error_timeseries ** 2)
                final_r2 = self._calculate_r2(y, predictions)
                training_time = time.time() - start_time
                
                # Store result
                results[method] = IdentificationResult(
                    method_name=method.replace('_', ' ').title(),
                    parameters=result['parameters'],
                    predictions=predictions,
                    error_timeseries=error_timeseries,
                    final_loss=float(final_loss),
                    final_r2=float(final_r2),
                    training_time=training_time,
                    convergence_info=result['convergence'],
                    model=result['model']
                )
                
                print(f"  ✓ Completed in {training_time:.2f}s, R² = {final_r2:.4f}")
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                results[method] = None
        
        return results
    
    def run_all_methods(self, u: jnp.ndarray, y: jnp.ndarray,
                       **kwargs) -> Dict[str, IdentificationResult]:
        """
        Run all available system identification methods.
        
        Args:
            u: Input voltage [V]
            y: Output measurements [current, velocity]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with results for all methods
        """
        print("🚀 COMPREHENSIVE SYSTEM IDENTIFICATION")
        print("=" * 80)
        print(f"Input shape: {u.shape}, Output shape: {y.shape}")
        print("=" * 80)
        
        all_results = {}
        
        # Run all method categories
        all_results.update(self.run_nonlinear_optimization_methods(u, y, **kwargs))
        all_results.update(self.run_bayesian_methods(u, y, **kwargs))
        all_results.update(self.run_gp_surrogate_methods(u, y, **kwargs))
        all_results.update(self.run_state_space_methods(u, y, **kwargs))
        all_results.update(self.run_original_dynax_methods(u, y, **kwargs))
        
        # Filter out failed methods
        successful_results = {k: v for k, v in all_results.items() if v is not None}
        
        print("\n" + "=" * 80)
        print("SUMMARY OF ALL METHODS")
        print("=" * 80)
        
        # Sort by R² score
        sorted_results = sorted(successful_results.items(), 
                              key=lambda x: x[1].final_r2, reverse=True)
        
        for i, (method_name, result) in enumerate(sorted_results, 1):
            print(f"{i:2d}. {result.method_name:30s} | "
                  f"R² = {result.final_r2:8.4f} | "
                  f"Loss = {result.final_loss:.6f} | "
                  f"Time = {result.training_time:.2f}s")
        
        self.results = successful_results
        return successful_results
    
    def _calculate_r2(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
        """Calculate R² coefficient of determination."""
        ss_res = jnp.sum((y_true - y_pred) ** 2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return float(r2)
    
    def generate_comprehensive_report(self, output_dir: str = "results") -> Dict[str, Any]:
        """
        Generate comprehensive report of all results.
        
        Args:
            output_dir: Output directory for reports
            
        Returns:
            Dictionary with report data
        """
        if not self.results:
            raise ValueError("No results available. Run identification methods first.")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Sort results by R² score
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1].final_r2, reverse=True)
        
        # Generate report data
        report_data = {
            'summary': {
                'total_methods': len(self.results),
                'best_method': sorted_results[0][1].method_name,
                'best_r2': sorted_results[0][1].final_r2,
                'best_loss': sorted_results[0][1].final_loss
            },
            'results': {}
        }
        
        for method_name, result in sorted_results:
            report_data['results'][method_name] = {
                'method_name': result.method_name,
                'parameters': {k: float(v) if isinstance(v, (int, float, jnp.ndarray)) 
                             else str(v) for k, v in result.parameters.items()},
                'final_loss': result.final_loss,
                'final_r2': result.final_r2,
                'training_time': result.training_time,
                'convergence_info': result.convergence_info
            }
        
        # Save report
        import json
        with open(output_path / "comprehensive_report.json", 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate summary text
        with open(output_path / "summary.txt", 'w') as f:
            f.write("COMPREHENSIVE SYSTEM IDENTIFICATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total methods tested: {len(self.results)}\n")
            f.write(f"Best method: {sorted_results[0][1].method_name}\n")
            f.write(f"Best R² score: {sorted_results[0][1].final_r2:.6f}\n")
            f.write(f"Best loss: {sorted_results[0][1].final_loss:.8f}\n\n")
            
            f.write("Method Ranking (by R² score):\n")
            f.write("-" * 50 + "\n")
            for i, (method_name, result) in enumerate(sorted_results, 1):
                f.write(f"{i:2d}. {result.method_name:30s} | "
                       f"R² = {result.final_r2:8.4f} | "
                       f"Loss = {result.final_loss:.6f} | "
                       f"Time = {result.training_time:.2f}s\n")
        
        print(f"\n📊 Comprehensive report generated in: {output_dir}")
        print(f"   - comprehensive_report.json: Complete results")
        print(f"   - summary.txt: Human-readable summary")
        
        return report_data


def run_comprehensive_identification(u: jnp.ndarray, y: jnp.ndarray,
                                   **kwargs) -> Dict[str, IdentificationResult]:
    """
    Run comprehensive system identification with all methods.
    
    Args:
        u: Input voltage [V]
        y: Output measurements [current, velocity]
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with results for all methods
    """
    identifier = ComprehensiveSystemIdentifier()
    return identifier.run_all_methods(u, y, **kwargs)
