"""
Test-Driven Development for Data Analysis Module.

This module implements comprehensive tests for the data analysis functionality,
following TDD principles.

Author: Research Team
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings
from tests.conftest import MathematicalTestMixin, PhysicalConstraintMixin


class TestLoudspeakerMeasurement(MathematicalTestMixin, PhysicalConstraintMixin):
    """Test suite for LoudspeakerMeasurement dataclass."""
    
    def test_measurement_initialization(self):
        """Test that LoudspeakerMeasurement can be initialized correctly."""
        from src.data_analysis import LoudspeakerMeasurement
        
        # Test data
        voltage = jnp.array([1.0, 2.0, 3.0])
        current = jnp.array([0.1, 0.2, 0.3])
        displacement = jnp.array([0.001, 0.002, 0.003])
        velocity = jnp.array([0.01, 0.02, 0.03])
        sample_rate = 48000.0
        excitation_type = "pink_noise"
        voltage_rms = 2.0
        frequency_range = (5.0, 2000.0)
        
        measurement = LoudspeakerMeasurement(
            voltage=voltage,
            current=current,
            displacement=displacement,
            velocity=velocity,
            sample_rate=sample_rate,
            duration=0.0,  # Will be computed
            excitation_type=excitation_type,
            voltage_rms=voltage_rms,
            frequency_range=frequency_range
        )
        
        # Test basic properties
        assert measurement.sample_rate == sample_rate
        assert measurement.excitation_type == excitation_type
        assert measurement.voltage_rms == voltage_rms
        assert measurement.frequency_range == frequency_range
        
        # Test computed properties
        assert measurement.n_samples == 3
        assert measurement.duration == 3 / sample_rate
        assert jnp.allclose(measurement.time_vector, jnp.array([0, 1, 2]) / sample_rate)
    
    def test_measurement_validation(self):
        """Test that measurement data is validated correctly."""
        from src.data_analysis import LoudspeakerMeasurement
        
        # Test with mismatched array lengths
        voltage = jnp.array([1.0, 2.0, 3.0])
        current = jnp.array([0.1, 0.2])  # Different length
        
        with pytest.raises(ValueError, match="Inconsistent signal lengths"):
            LoudspeakerMeasurement(
                voltage=voltage,
                current=current,
                displacement=jnp.array([0.001, 0.002, 0.003]),
                velocity=jnp.array([0.01, 0.02, 0.03]),
                sample_rate=48000.0,
                duration=0.0,
                excitation_type="pink_noise",
                voltage_rms=2.0,
                frequency_range=(5.0, 2000.0)
            )
    
    def test_rms_calculation(self):
        """Test RMS calculation in post_init."""
        from src.data_analysis import LoudspeakerMeasurement
        
        # Create measurement with known RMS
        voltage = jnp.array([1.0, -1.0, 2.0, -2.0])
        current = jnp.array([0.1, -0.1, 0.2, -0.2])
        displacement = jnp.array([0.001, -0.001, 0.002, -0.002])
        velocity = jnp.array([0.01, -0.01, 0.02, -0.02])
        
        measurement = LoudspeakerMeasurement(
            voltage=voltage,
            current=current,
            displacement=displacement,
            velocity=velocity,
            sample_rate=48000.0,
            duration=0.0,
            excitation_type="pink_noise",
            voltage_rms=2.0,
            frequency_range=(5.0, 2000.0)
        )
        
        # Test computed RMS values
        expected_voltage_rms = jnp.sqrt(jnp.mean(voltage**2))
        expected_current_rms = jnp.sqrt(jnp.mean(current**2))
        
        assert jnp.isclose(measurement.voltage_rms_computed, expected_voltage_rms)
        assert jnp.isclose(measurement.current_rms_computed, expected_current_rms)
    
    def test_statistics_calculation(self):
        """Test statistical calculations."""
        from src.data_analysis import LoudspeakerMeasurement
        
        # Create measurement with known statistics
        displacement = jnp.array([0.001, -0.001, 0.002, -0.002, 0.0])
        velocity = jnp.array([0.01, -0.01, 0.02, -0.02, 0.0])
        
        measurement = LoudspeakerMeasurement(
            voltage=jnp.array([1.0, -1.0, 2.0, -2.0, 0.0]),
            current=jnp.array([0.1, -0.1, 0.2, -0.2, 0.0]),
            displacement=displacement,
            velocity=velocity,
            sample_rate=48000.0,
            duration=0.0,
            excitation_type="pink_noise",
            voltage_rms=2.0,
            frequency_range=(5.0, 2000.0)
        )
        
        # Test displacement statistics
        expected_std = jnp.std(displacement)
        expected_peak = jnp.max(jnp.abs(displacement))
        
        assert jnp.isclose(measurement.displacement_std, expected_std)
        assert jnp.isclose(measurement.displacement_peak, expected_peak)
        
        # Test velocity statistics
        expected_vel_std = jnp.std(velocity)
        expected_vel_peak = jnp.max(jnp.abs(velocity))
        
        assert jnp.isclose(measurement.velocity_std, expected_vel_std)
        assert jnp.isclose(measurement.velocity_peak, expected_vel_peak)


class TestDTULoudspeakerAnalyzer(MathematicalTestMixin, PhysicalConstraintMixin):
    """Test suite for DTULoudspeakerAnalyzer class."""
    
    @pytest.fixture
    def mock_mat_data(self):
        """Mock .mat file data for testing."""
        return {
            'voltage': np.random.randn(1000),
            'current': np.random.randn(1000),
            'displacement': np.random.randn(1000),
            'velocity': np.random.randn(1000),
            'sample_rate': np.array([[48000.0]])
        }
    
    def test_analyzer_initialization(self):
        """Test that analyzer can be initialized correctly."""
        from src.data_analysis import DTULoudspeakerAnalyzer
        
        analyzer = DTULoudspeakerAnalyzer("test_data_path")
        
        assert analyzer.data_path == Path("test_data_path")
        assert analyzer.measurements == {}
        assert analyzer.analysis_results == {}
    
    @patch('src.data_analysis.loadmat')
    def test_load_mat_file(self, mock_loadmat, mock_mat_data):
        """Test loading of .mat files."""
        from src.data_analysis import DTULoudspeakerAnalyzer
        
        mock_loadmat.return_value = mock_mat_data
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        measurement = analyzer._load_mat_file(Path("test_file.mat"), "test_measurement")
        
        # Test that measurement was created correctly
        assert measurement.sample_rate == 48000.0
        assert measurement.excitation_type == "pink_noise"
        assert measurement.voltage_rms == 8.0
        assert measurement.frequency_range == (5.0, 2000.0)
        assert measurement.n_samples == 1000
    
    @patch('src.data_analysis.loadmat')
    def test_load_mat_file_error_handling(self, mock_loadmat):
        """Test error handling in _load_mat_file."""
        from src.data_analysis import DTULoudspeakerAnalyzer
        
        mock_loadmat.side_effect = Exception("File not found")
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        
        with pytest.raises(RuntimeError, match="Failed to load"):
            analyzer._load_mat_file(Path("nonexistent.mat"), "test")
    
    def test_analyze_signal_statistics(self, synthetic_loudspeaker_data):
        """Test signal statistical analysis."""
        from src.data_analysis import DTULoudspeakerAnalyzer, LoudspeakerMeasurement
        
        # Create measurement from synthetic data
        measurement = LoudspeakerMeasurement(
            voltage=synthetic_loudspeaker_data['voltage'],
            current=synthetic_loudspeaker_data['current'],
            displacement=synthetic_loudspeaker_data['displacement'],
            velocity=synthetic_loudspeaker_data['velocity'],
            sample_rate=synthetic_loudspeaker_data['sample_rate'],
            duration=0.0,
            excitation_type="pink_noise",
            voltage_rms=2.0,
            frequency_range=(5.0, 2000.0)
        )
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        results = analyzer.analyze_signal_statistics(measurement)
        
        # Test that all signals are analyzed
        expected_signals = ['voltage', 'current', 'displacement', 'velocity']
        for signal in expected_signals:
            assert signal in results
            assert 'mean' in results[signal]
            assert 'std' in results[signal]
            assert 'rms' in results[signal]
            assert 'peak' in results[signal]
            assert 'crest_factor' in results[signal]
            assert 'skewness' in results[signal]
            assert 'kurtosis' in results[signal]
        
        # Test cross-correlations
        assert 'cross_correlations' in results
        assert 'voltage_current' in results['cross_correlations']
        
        # Test velocity consistency
        assert 'velocity_consistency' in results
        assert 'mse' in results['velocity_consistency']
        assert 'correlation' in results['velocity_consistency']
    
    def test_skewness_calculation(self):
        """Test skewness calculation."""
        from src.data_analysis import DTULoudspeakerAnalyzer
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        
        # Test with symmetric data (should have skewness ≈ 0)
        symmetric_data = jnp.array([-2, -1, 0, 1, 2])
        skewness = analyzer._compute_skewness(symmetric_data)
        assert jnp.isclose(skewness, 0.0, atol=1e-10)
        
        # Test with asymmetric data
        asymmetric_data = jnp.array([1, 2, 3, 4, 5])
        skewness = analyzer._compute_skewness(asymmetric_data)
        assert skewness > 0  # Should be positive for right-skewed data
    
    def test_kurtosis_calculation(self):
        """Test kurtosis calculation."""
        from src.data_analysis import DTULoudspeakerAnalyzer
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        
        # Test with normal data (should have kurtosis ≈ 0)
        normal_data = jax.random.normal(jax.random.PRNGKey(42), (1000,))
        kurtosis = analyzer._compute_kurtosis(normal_data)
        assert jnp.isclose(kurtosis, 0.0, atol=0.1)  # Allow some tolerance
    
    def test_cross_correlations(self):
        """Test cross-correlation calculation."""
        from src.data_analysis import DTULoudspeakerAnalyzer
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        
        # Test with perfectly correlated signals
        signal1 = jnp.array([1.0, 2.0, 3.0, 4.0])
        signal2 = 2.0 * signal1  # Perfectly correlated
        
        signals = {'signal1': signal1, 'signal2': signal2}
        correlations = analyzer._compute_cross_correlations(signals)
        
        assert 'signal1_signal2' in correlations
        assert jnp.isclose(correlations['signal1_signal2'], 1.0)
    
    @patch('src.data_analysis.signal.welch')
    def test_analyze_frequency_content(self, mock_welch):
        """Test frequency domain analysis."""
        from src.data_analysis import DTULoudspeakerAnalyzer, LoudspeakerMeasurement
        
        # Mock Welch's method
        mock_f = jnp.linspace(0, 24000, 1000)
        mock_psd = jnp.ones(1000)
        mock_welch.return_value = (mock_f, mock_psd)
        
        # Create test measurement
        measurement = LoudspeakerMeasurement(
            voltage=jnp.random.randn(1000),
            current=jnp.random.randn(1000),
            displacement=jnp.random.randn(1000),
            velocity=jnp.random.randn(1000),
            sample_rate=48000.0,
            duration=0.0,
            excitation_type="pink_noise",
            voltage_rms=2.0,
            frequency_range=(5.0, 2000.0)
        )
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        results = analyzer.analyze_frequency_content(measurement)
        
        # Test that all signals are analyzed
        expected_signals = ['voltage', 'current', 'displacement', 'velocity']
        for signal in expected_signals:
            assert signal in results
            assert 'frequencies' in results[signal]
            assert 'psd' in results[signal]
            assert 'total_power' in results[signal]
            assert 'peak_frequency' in results[signal]
            assert 'bandwidth_3db' in results[signal]
        
        # Test transfer functions
        assert 'transfer_functions' in results
        assert 'voltage_to_current' in results['transfer_functions']
        assert 'voltage_to_displacement' in results['transfer_functions']
        assert 'voltage_to_velocity' in results['transfer_functions']
        
        # Test coherence
        assert 'coherence' in results
    
    def test_3db_bandwidth_calculation(self):
        """Test 3dB bandwidth calculation."""
        from src.data_analysis import DTULoudspeakerAnalyzer
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        
        # Test with simple peak
        f = jnp.linspace(0, 1000, 100)
        psd = jnp.zeros(100)
        psd = psd.at[50].set(1.0)  # Peak at middle frequency
        
        bandwidth = analyzer._compute_3db_bandwidth(f, psd)
        assert bandwidth == 0.0  # Single point peak has zero bandwidth
        
        # Test with broader peak
        psd = jnp.exp(-((f - 500) / 50)**2)  # Gaussian peak
        bandwidth = analyzer._compute_3db_bandwidth(f, psd)
        assert bandwidth > 0  # Should have positive bandwidth
    
    def test_assess_data_quality(self, synthetic_loudspeaker_data):
        """Test data quality assessment."""
        from src.data_analysis import DTULoudspeakerAnalyzer, LoudspeakerMeasurement
        
        # Create measurement
        measurement = LoudspeakerMeasurement(
            voltage=synthetic_loudspeaker_data['voltage'],
            current=synthetic_loudspeaker_data['current'],
            displacement=synthetic_loudspeaker_data['displacement'],
            velocity=synthetic_loudspeaker_data['velocity'],
            sample_rate=synthetic_loudspeaker_data['sample_rate'],
            duration=0.0,
            excitation_type="pink_noise",
            voltage_rms=2.0,
            frequency_range=(5.0, 2000.0)
        )
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        results = analyzer.assess_data_quality(measurement)
        
        # Test SNR estimation
        assert 'snr_estimation' in results
        expected_signals = ['voltage', 'current', 'displacement', 'velocity']
        for signal in expected_signals:
            assert signal in results['snr_estimation']
            assert 'snr_db' in results['snr_estimation'][signal]
            assert 'noise_floor' in results['snr_estimation'][signal]
            assert 'signal_power' in results['snr_estimation'][signal]
        
        # Test excitation quality
        assert 'excitation_quality' in results
        assert 'frequency_coverage_ratio' in results['excitation_quality']
        assert 'excitation_adequacy' in results['excitation_quality']
        
        # Test linearity assessment
        assert 'linearity_assessment' in results
        assert 'mean_coherence' in results['linearity_assessment']
        assert 'linearity_score' in results['linearity_assessment']
        
        # Test completeness
        assert 'completeness' in results
        for signal in expected_signals:
            assert signal in results['completeness']
            assert 'is_valid' in results['completeness'][signal]
    
    def test_generate_comprehensive_report(self, synthetic_loudspeaker_data):
        """Test comprehensive report generation."""
        from src.data_analysis import DTULoudspeakerAnalyzer, LoudspeakerMeasurement
        
        # Create measurement
        measurement = LoudspeakerMeasurement(
            voltage=synthetic_loudspeaker_data['voltage'],
            current=synthetic_loudspeaker_data['current'],
            displacement=synthetic_loudspeaker_data['displacement'],
            velocity=synthetic_loudspeaker_data['velocity'],
            sample_rate=synthetic_loudspeaker_data['sample_rate'],
            duration=0.0,
            excitation_type="pink_noise",
            voltage_rms=2.0,
            frequency_range=(5.0, 2000.0)
        )
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        analyzer.measurements['test_measurement'] = measurement
        
        report = analyzer.generate_comprehensive_report('test_measurement')
        
        # Test that report contains expected sections
        assert 'COMPREHENSIVE LOUDSPEAKER MEASUREMENT ANALYSIS' in report
        assert 'SIGNAL STATISTICS' in report
        assert 'DATA QUALITY ASSESSMENT' in report
        assert 'RECOMMENDATIONS' in report
        
        # Test that report contains measurement info
        assert 'test_measurement' in report
        assert 'pink_noise' in report
        assert '48000' in report  # sample rate
    
    def test_generate_report_nonexistent_measurement(self):
        """Test error handling for nonexistent measurement."""
        from src.data_analysis import DTULoudspeakerAnalyzer
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        
        with pytest.raises(ValueError, match="Measurement 'nonexistent' not found"):
            analyzer.generate_comprehensive_report('nonexistent')


class TestDataAnalysisIntegration:
    """Integration tests for data analysis module."""
    
    def test_end_to_end_analysis(self, synthetic_loudspeaker_data):
        """Test complete end-to-end analysis workflow."""
        from src.data_analysis import DTULoudspeakerAnalyzer, LoudspeakerMeasurement
        
        # Create measurement
        measurement = LoudspeakerMeasurement(
            voltage=synthetic_loudspeaker_data['voltage'],
            current=synthetic_loudspeaker_data['current'],
            displacement=synthetic_loudspeaker_data['displacement'],
            velocity=synthetic_loudspeaker_data['velocity'],
            sample_rate=synthetic_loudspeaker_data['sample_rate'],
            duration=0.0,
            excitation_type="pink_noise",
            voltage_rms=2.0,
            frequency_range=(5.0, 2000.0)
        )
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        
        # Perform all analyses
        stats_results = analyzer.analyze_signal_statistics(measurement)
        freq_results = analyzer.analyze_frequency_content(measurement)
        quality_results = analyzer.assess_data_quality(measurement)
        
        # Test that all analyses completed successfully
        assert len(stats_results) > 0
        assert len(freq_results) > 0
        assert len(quality_results) > 0
        
        # Test that results are consistent
        assert 'voltage' in stats_results
        assert 'voltage' in freq_results
        assert 'voltage' in quality_results['snr_estimation']
    
    @given(st.floats(min_value=0.1, max_value=10.0))
    def test_signal_analysis_properties(self, amplitude):
        """Test that signal analysis satisfies mathematical properties."""
        from src.data_analysis import DTULoudspeakerAnalyzer, LoudspeakerMeasurement
        
        # Create test signal
        n_samples = 1000
        t = jnp.linspace(0, 1, n_samples)
        signal = amplitude * jnp.sin(2 * jnp.pi * 10 * t)
        
        measurement = LoudspeakerMeasurement(
            voltage=signal,
            current=signal * 0.1,
            displacement=signal * 0.001,
            velocity=signal * 0.01,
            sample_rate=1000.0,
            duration=0.0,
            excitation_type="sine",
            voltage_rms=amplitude / jnp.sqrt(2),
            frequency_range=(5.0, 1000.0)
        )
        
        analyzer = DTULoudspeakerAnalyzer("test_path")
        results = analyzer.analyze_signal_statistics(measurement)
        
        # Test mathematical properties
        # RMS should be close to amplitude/sqrt(2) for sine wave
        expected_rms = amplitude / jnp.sqrt(2)
        assert jnp.isclose(results['voltage']['rms'], expected_rms, rtol=0.1)
        
        # Peak should equal amplitude
        assert jnp.isclose(results['voltage']['peak'], amplitude)
        
        # Crest factor should be sqrt(2) for sine wave
        assert jnp.isclose(results['voltage']['crest_factor'], jnp.sqrt(2), rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
