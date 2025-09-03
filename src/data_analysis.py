"""
Comprehensive data analysis module for DTU loudspeaker dataset.

This module provides rigorous analysis of the DTU loudspeaker measurements,
including statistical characterization, frequency domain analysis, and
data quality assessment for system identification.

Author: Research Team
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy import signal
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pandas as pd
from dataclasses import dataclass
import warnings

@dataclass
class LoudspeakerMeasurement:
    """Container for loudspeaker measurement data with metadata."""
    voltage: jnp.ndarray
    current: jnp.ndarray
    displacement: jnp.ndarray
    velocity: jnp.ndarray
    sample_rate: float
    duration: float
    excitation_type: str
    voltage_rms: float
    frequency_range: Tuple[float, float]
    
    def __post_init__(self):
        """Validate and compute derived quantities."""
        self.n_samples = len(self.voltage)
        self.time_vector = jnp.arange(self.n_samples) / self.sample_rate
        self.duration = self.n_samples / self.sample_rate
        
        # Compute RMS values for validation
        self.voltage_rms_computed = float(jnp.sqrt(jnp.mean(self.voltage**2)))
        self.current_rms_computed = float(jnp.sqrt(jnp.mean(self.current**2)))
        
        # Compute displacement statistics
        self.displacement_std = float(jnp.std(self.displacement))
        self.displacement_peak = float(jnp.max(jnp.abs(self.displacement)))
        
        # Compute velocity statistics
        self.velocity_std = float(jnp.std(self.velocity))
        self.velocity_peak = float(jnp.max(jnp.abs(self.velocity)))


class DTULoudspeakerAnalyzer:
    """
    Comprehensive analyzer for DTU loudspeaker dataset.
    
    This class provides methods for:
    - Loading and validating measurement data
    - Statistical characterization of signals
    - Frequency domain analysis
    - Data quality assessment
    - Cross-validation between different measurement protocols
    """
    
    def __init__(self, data_path: str = "data/raw/loudspeaker-datasets"):
        """Initialize analyzer with path to DTU dataset."""
        self.data_path = Path(data_path)
        self.measurements: Dict[str, LoudspeakerMeasurement] = {}
        self.analysis_results: Dict[str, Dict] = {}
        
    def load_pink_noise_measurements(self) -> Dict[str, LoudspeakerMeasurement]:
        """
        Load all pink noise measurements from DTU dataset.
        
        Returns:
            Dictionary mapping measurement names to LoudspeakerMeasurement objects.
        """
        pink_noise_files = [
            "dtu_34871-NL-Transducers/ExpC/pinknoise_5Hz-2000Hz_8Vrms.mat",
            "dtu_34871-NL-Transducers/ExpD/pinknoise_5Hz-2000Hz_8Vrms.mat"
        ]
        
        measurements = {}
        
        for file_path in pink_noise_files:
            full_path = self.data_path / file_path
            if full_path.exists():
                measurement_name = file_path.split('/')[-2]  # ExpC or ExpD
                measurements[measurement_name] = self._load_mat_file(full_path, measurement_name)
            else:
                warnings.warn(f"Pink noise file not found: {full_path}")
                
        self.measurements.update(measurements)
        return measurements
    
    def _load_mat_file(self, file_path: Path, measurement_name: str) -> LoudspeakerMeasurement:
        """Load and validate a single .mat file."""
        try:
            mat_contents = loadmat(str(file_path))
            
            # Extract data with proper error handling
            voltage = jnp.array(mat_contents["voltage"][0], dtype=jnp.float32)
            current = jnp.array(mat_contents["current"][0], dtype=jnp.float32)
            displacement = jnp.array(mat_contents["displacement"][0], dtype=jnp.float32)
            velocity = jnp.array(mat_contents["velocity"][0], dtype=jnp.float32)
            sample_rate = float(mat_contents["sample_rate"][0, 0])
            
            # Validate data consistency
            n_samples = len(voltage)
            if not all(len(signal) == n_samples for signal in [current, displacement, velocity]):
                raise ValueError("Inconsistent signal lengths in measurement data")
            
            # Determine excitation parameters from filename
            if "pinknoise" in file_path.name:
                excitation_type = "pink_noise"
                voltage_rms = 8.0  # From filename
                frequency_range = (5.0, 2000.0)  # From filename
            else:
                excitation_type = "unknown"
                voltage_rms = float(jnp.sqrt(jnp.mean(voltage**2)))
                frequency_range = (0.0, sample_rate/2)
            
            return LoudspeakerMeasurement(
                voltage=voltage,
                current=current,
                displacement=displacement,
                velocity=velocity,
                sample_rate=sample_rate,
                duration=n_samples / sample_rate,
                excitation_type=excitation_type,
                voltage_rms=voltage_rms,
                frequency_range=frequency_range
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {e}")
    
    def analyze_signal_statistics(self, measurement: LoudspeakerMeasurement) -> Dict:
        """
        Perform comprehensive statistical analysis of measurement signals.
        
        Args:
            measurement: LoudspeakerMeasurement object to analyze
            
        Returns:
            Dictionary containing statistical analysis results
        """
        results = {}
        
        # Time domain statistics
        signals = {
            'voltage': measurement.voltage,
            'current': measurement.current,
            'displacement': measurement.displacement,
            'velocity': measurement.velocity
        }
        
        for name, signal in signals.items():
            results[name] = {
                'mean': float(jnp.mean(signal)),
                'std': float(jnp.std(signal)),
                'rms': float(jnp.sqrt(jnp.mean(signal**2))),
                'peak': float(jnp.max(jnp.abs(signal))),
                'crest_factor': float(jnp.max(jnp.abs(signal)) / jnp.sqrt(jnp.mean(signal**2))),
                'skewness': float(self._compute_skewness(signal)),
                'kurtosis': float(self._compute_kurtosis(signal))
            }
        
        # Cross-correlation analysis
        results['cross_correlations'] = self._compute_cross_correlations(signals)
        
        # Displacement-velocity relationship validation
        velocity_computed = jnp.gradient(measurement.displacement, 1/measurement.sample_rate)
        velocity_error = jnp.mean((measurement.velocity - velocity_computed)**2)
        results['velocity_consistency'] = {
            'mse': float(velocity_error),
            'correlation': float(jnp.corrcoef(measurement.velocity, velocity_computed)[0, 1])
        }
        
        return results
    
    def _compute_skewness(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute skewness of signal."""
        mean_x = jnp.mean(x)
        std_x = jnp.std(x)
        return jnp.mean(((x - mean_x) / std_x) ** 3)
    
    def _compute_kurtosis(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute kurtosis of signal."""
        mean_x = jnp.mean(x)
        std_x = jnp.std(x)
        return jnp.mean(((x - mean_x) / std_x) ** 4) - 3
    
    def _compute_cross_correlations(self, signals: Dict[str, jnp.ndarray]) -> Dict:
        """Compute cross-correlations between all signal pairs."""
        correlations = {}
        signal_names = list(signals.keys())
        
        for i, name1 in enumerate(signal_names):
            for j, name2 in enumerate(signal_names[i+1:], i+1):
                corr = jnp.corrcoef(signals[name1], signals[name2])[0, 1]
                correlations[f"{name1}_{name2}"] = float(corr)
        
        return correlations
    
    def analyze_frequency_content(self, measurement: LoudspeakerMeasurement, 
                                nperseg: int = 2**14, overlap: float = 0.75) -> Dict:
        """
        Perform frequency domain analysis using Welch's method.
        
        Args:
            measurement: LoudspeakerMeasurement object
            nperseg: Length of each segment for PSD estimation
            overlap: Overlap fraction between segments
            
        Returns:
            Dictionary containing frequency domain analysis results
        """
        results = {}
        
        # Compute power spectral densities
        signals = {
            'voltage': measurement.voltage,
            'current': measurement.current,
            'displacement': measurement.displacement,
            'velocity': measurement.velocity
        }
        
        for name, signal in signals.items():
            f, psd = signal.welch(signal, fs=measurement.sample_rate, 
                                nperseg=nperseg, noverlap=int(nperseg * overlap))
            
            results[name] = {
                'frequencies': f,
                'psd': psd,
                'total_power': float(jnp.trapz(psd, f)),
                'peak_frequency': float(f[jnp.argmax(psd)]),
                'bandwidth_3db': self._compute_3db_bandwidth(f, psd)
            }
        
        # Compute transfer functions
        results['transfer_functions'] = self._compute_transfer_functions(
            measurement, nperseg, overlap)
        
        # Compute coherence functions
        results['coherence'] = self._compute_coherence_functions(
            measurement, nperseg, overlap)
        
        return results
    
    def _compute_3db_bandwidth(self, f: jnp.ndarray, psd: jnp.ndarray) -> float:
        """Compute 3dB bandwidth of PSD."""
        peak_idx = jnp.argmax(psd)
        peak_psd = psd[peak_idx]
        threshold = peak_psd / 2  # 3dB down
        
        # Find frequencies where PSD crosses threshold
        above_threshold = psd > threshold
        if jnp.sum(above_threshold) < 2:
            return 0.0
        
        indices = jnp.where(above_threshold)[0]
        return float(f[indices[-1]] - f[indices[0]])
    
    def _compute_transfer_functions(self, measurement: LoudspeakerMeasurement,
                                  nperseg: int, overlap: float) -> Dict:
        """Compute transfer functions between voltage and other signals."""
        results = {}
        
        # Voltage to current (electrical impedance)
        f, h_vi = signal.coherence(measurement.voltage, measurement.current,
                                 fs=measurement.sample_rate, nperseg=nperseg,
                                 noverlap=int(nperseg * overlap))
        results['voltage_to_current'] = {'frequencies': f, 'coherence': h_vi}
        
        # Voltage to displacement
        f, h_vx = signal.coherence(measurement.voltage, measurement.displacement,
                                 fs=measurement.sample_rate, nperseg=nperseg,
                                 noverlap=int(nperseg * overlap))
        results['voltage_to_displacement'] = {'frequencies': f, 'coherence': h_vx}
        
        # Voltage to velocity
        f, h_vv = signal.coherence(measurement.voltage, measurement.velocity,
                                 fs=measurement.sample_rate, nperseg=nperseg,
                                 noverlap=int(nperseg * overlap))
        results['voltage_to_velocity'] = {'frequencies': f, 'coherence': h_vv}
        
        return results
    
    def _compute_coherence_functions(self, measurement: LoudspeakerMeasurement,
                                   nperseg: int, overlap: float) -> Dict:
        """Compute coherence functions between all signal pairs."""
        results = {}
        signals = {
            'voltage': measurement.voltage,
            'current': measurement.current,
            'displacement': measurement.displacement,
            'velocity': measurement.velocity
        }
        
        signal_names = list(signals.keys())
        for i, name1 in enumerate(signal_names):
            for j, name2 in enumerate(signal_names[i+1:], i+1):
                f, coherence = signal.coherence(signals[name1], signals[name2],
                                              fs=measurement.sample_rate,
                                              nperseg=nperseg,
                                              noverlap=int(nperseg * overlap))
                results[f"{name1}_{name2}"] = {
                    'frequencies': f,
                    'coherence': coherence,
                    'mean_coherence': float(jnp.mean(coherence))
                }
        
        return results
    
    def assess_data_quality(self, measurement: LoudspeakerMeasurement) -> Dict:
        """
        Assess data quality for system identification.
        
        Returns:
            Dictionary containing data quality metrics and recommendations
        """
        results = {}
        
        # Signal-to-noise ratio estimation
        results['snr_estimation'] = self._estimate_snr(measurement)
        
        # Excitation quality assessment
        results['excitation_quality'] = self._assess_excitation_quality(measurement)
        
        # Linearity assessment
        results['linearity_assessment'] = self._assess_linearity(measurement)
        
        # Data completeness
        results['completeness'] = self._assess_completeness(measurement)
        
        return results
    
    def _estimate_snr(self, measurement: LoudspeakerMeasurement) -> Dict:
        """Estimate signal-to-noise ratio for each signal."""
        snr_results = {}
        
        signals = {
            'voltage': measurement.voltage,
            'current': measurement.current,
            'displacement': measurement.displacement,
            'velocity': measurement.velocity
        }
        
        for name, signal in signals.items():
            # Simple SNR estimation using high-frequency noise floor
            f, psd = signal.welch(signal, fs=measurement.sample_rate)
            
            # Assume noise floor is in highest frequency band
            noise_floor = jnp.mean(psd[-len(psd)//10:])
            signal_power = jnp.max(psd)
            snr_db = 10 * jnp.log10(signal_power / noise_floor)
            
            snr_results[name] = {
                'snr_db': float(snr_db),
                'noise_floor': float(noise_floor),
                'signal_power': float(signal_power)
            }
        
        return snr_results
    
    def _assess_excitation_quality(self, measurement: LoudspeakerMeasurement) -> Dict:
        """Assess quality of excitation signal for system identification."""
        # For pink noise, check frequency content
        f, psd = signal.welch(measurement.voltage, fs=measurement.sample_rate)
        
        # Check if excitation covers the expected frequency range
        expected_range = measurement.frequency_range
        f_mask = (f >= expected_range[0]) & (f <= expected_range[1])
        excitation_power = jnp.trapz(psd[f_mask], f[f_mask])
        total_power = jnp.trapz(psd, f)
        coverage_ratio = excitation_power / total_power
        
        return {
            'frequency_coverage_ratio': float(coverage_ratio),
            'expected_range': expected_range,
            'actual_range': (float(f[1]), float(f[-1])),
            'excitation_adequacy': coverage_ratio > 0.8
        }
    
    def _assess_linearity(self, measurement: LoudspeakerMeasurement) -> Dict:
        """Assess linearity of the loudspeaker response."""
        # Compute coherence between voltage and current
        f, coherence = signal.coherence(measurement.voltage, measurement.current,
                                      fs=measurement.sample_rate)
        
        # High coherence indicates linear relationship
        mean_coherence = jnp.mean(coherence)
        min_coherence = jnp.min(coherence)
        
        return {
            'mean_coherence': float(mean_coherence),
            'min_coherence': float(min_coherence),
            'linearity_score': float(mean_coherence),
            'is_linear': mean_coherence > 0.8
        }
    
    def _assess_completeness(self, measurement: LoudspeakerMeasurement) -> Dict:
        """Assess completeness of measurement data."""
        signals = {
            'voltage': measurement.voltage,
            'current': measurement.current,
            'displacement': measurement.displacement,
            'velocity': measurement.velocity
        }
        
        completeness = {}
        for name, signal in signals.items():
            # Check for NaN or infinite values
            has_nan = jnp.any(jnp.isnan(signal))
            has_inf = jnp.any(jnp.isinf(signal))
            is_zero = jnp.all(signal == 0)
            
            completeness[name] = {
                'has_nan': bool(has_nan),
                'has_inf': bool(has_inf),
                'is_zero': bool(is_zero),
                'is_valid': not (has_nan or has_inf or is_zero)
            }
        
        return completeness
    
    def generate_comprehensive_report(self, measurement_name: str) -> str:
        """
        Generate a comprehensive analysis report for a measurement.
        
        Args:
            measurement_name: Name of the measurement to analyze
            
        Returns:
            Formatted report string
        """
        if measurement_name not in self.measurements:
            raise ValueError(f"Measurement '{measurement_name}' not found")
        
        measurement = self.measurements[measurement_name]
        
        # Perform all analyses
        stats = self.analyze_signal_statistics(measurement)
        freq_analysis = self.analyze_frequency_content(measurement)
        quality = self.assess_data_quality(measurement)
        
        # Generate report
        report = f"""
=== COMPREHENSIVE LOUDSPEAKER MEASUREMENT ANALYSIS ===
Measurement: {measurement_name}
Excitation: {measurement.excitation_type}
Duration: {measurement.duration:.2f} seconds
Sample Rate: {measurement.sample_rate} Hz
Voltage RMS: {measurement.voltage_rms:.2f} V

=== SIGNAL STATISTICS ===
"""
        
        for signal_name, signal_stats in stats.items():
            if isinstance(signal_stats, dict) and 'mean' in signal_stats:
                report += f"\n{signal_name.upper()}:"
                report += f"\n  Mean: {signal_stats['mean']:.6f}"
                report += f"\n  Std: {signal_stats['std']:.6f}"
                report += f"\n  RMS: {signal_stats['rms']:.6f}"
                report += f"\n  Peak: {signal_stats['peak']:.6f}"
                report += f"\n  Crest Factor: {signal_stats['crest_factor']:.2f}"
                report += f"\n  Skewness: {signal_stats['skewness']:.3f}"
                report += f"\n  Kurtosis: {signal_stats['kurtosis']:.3f}"
        
        report += f"\n\n=== DATA QUALITY ASSESSMENT ==="
        report += f"\nExcitation Quality: {'GOOD' if quality['excitation_quality']['excitation_adequacy'] else 'POOR'}"
        report += f"\nLinearity Score: {quality['linearity_assessment']['linearity_score']:.3f}"
        report += f"\nMean Coherence: {quality['linearity_assessment']['mean_coherence']:.3f}"
        
        report += f"\n\n=== RECOMMENDATIONS ==="
        if quality['linearity_assessment']['linearity_score'] < 0.8:
            report += "\n- Consider nonlinear system identification methods"
        if quality['excitation_quality']['frequency_coverage_ratio'] < 0.8:
            report += "\n- Excitation may not cover sufficient frequency range"
        
        return report


def main():
    """Demonstrate the data analysis capabilities."""
    analyzer = DTULoudspeakerAnalyzer()
    
    # Load pink noise measurements
    measurements = analyzer.load_pink_noise_measurements()
    
    if measurements:
        print(f"Loaded {len(measurements)} pink noise measurements")
        
        # Analyze each measurement
        for name, measurement in measurements.items():
            print(f"\nAnalyzing {name}...")
            report = analyzer.generate_comprehensive_report(name)
            print(report)
    else:
        print("No pink noise measurements found!")


if __name__ == "__main__":
    main()
