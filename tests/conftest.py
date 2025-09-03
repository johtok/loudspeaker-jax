"""
Comprehensive test configuration and utilities for TDD framework.

This module provides:
- Fixtures for test data generation
- JAX-specific testing utilities
- Mathematical property testing helpers
- Performance testing utilities
- Mock objects for external dependencies

Author: Research Team
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Any, Callable, Optional
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from hypothesis import strategies as st
from hypothesis import given, settings, example
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests


@dataclass
class TestConfig:
    """Configuration for test execution."""
    rtol: float = 1e-6
    atol: float = 1e-8
    max_iterations: int = 1000
    random_seed: int = 42
    enable_gpu: bool = False
    verbose: bool = False


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return TestConfig()


@pytest.fixture(scope="session")
def jax_config():
    """Configure JAX for testing."""
    # Set random seed for reproducibility
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_prng_impl", "rbg")
    
    # Enable compilation for performance tests
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    
    return jax.config


@pytest.fixture
def rng_key(test_config):
    """Provide reproducible random key for tests."""
    return jax.random.PRNGKey(test_config.random_seed)


@pytest.fixture
def rng_keys(rng_key):
    """Provide multiple random keys for tests."""
    return jax.random.split(rng_key, 10)


# ============================================================================
# Mathematical Testing Utilities
# ============================================================================

class MathematicalTestMixin:
    """Mixin class providing mathematical testing utilities."""
    
    @staticmethod
    def assert_close(a: jnp.ndarray, b: jnp.ndarray, 
                    rtol: float = 1e-6, atol: float = 1e-8) -> None:
        """Assert two arrays are close with specified tolerances."""
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    
    @staticmethod
    def assert_positive_definite(matrix: jnp.ndarray) -> None:
        """Assert matrix is positive definite."""
        eigenvals = jnp.linalg.eigvals(matrix)
        assert jnp.all(eigenvals > 0), f"Matrix not positive definite. Min eigenvalue: {jnp.min(eigenvals)}"
    
    @staticmethod
    def assert_symmetric(matrix: jnp.ndarray, rtol: float = 1e-10) -> None:
        """Assert matrix is symmetric."""
        np.testing.assert_allclose(matrix, matrix.T, rtol=rtol)
    
    @staticmethod
    def assert_orthogonal(matrix: jnp.ndarray, rtol: float = 1e-10) -> None:
        """Assert matrix is orthogonal."""
        identity = jnp.eye(matrix.shape[0])
        np.testing.assert_allclose(matrix @ matrix.T, identity, rtol=rtol)
    
    @staticmethod
    def assert_positive(x: jnp.ndarray) -> None:
        """Assert all elements are positive."""
        assert jnp.all(x > 0), f"Not all elements positive. Min value: {jnp.min(x)}"
    
    @staticmethod
    def assert_finite(x: jnp.ndarray) -> None:
        """Assert all elements are finite."""
        assert jnp.all(jnp.isfinite(x)), "Array contains non-finite values"


class PhysicalConstraintMixin:
    """Mixin class for testing physical constraints."""
    
    @staticmethod
    def assert_energy_conservation(initial_energy: float, final_energy: float, 
                                 tolerance: float = 1e-3) -> None:
        """Assert energy conservation within tolerance."""
        energy_change = abs(final_energy - initial_energy) / abs(initial_energy)
        assert energy_change < tolerance, f"Energy not conserved. Change: {energy_change:.2%}"
    
    @staticmethod
    def assert_causality(impulse_response: jnp.ndarray, 
                        sample_rate: float) -> None:
        """Assert system is causal (no response before t=0)."""
        # Check that response is zero before t=0
        assert jnp.all(impulse_response[:int(sample_rate * 0.001)] == 0), \
            "System is not causal"
    
    @staticmethod
    def assert_passivity(impedance: jnp.ndarray) -> None:
        """Assert system is passive (real part of impedance >= 0)."""
        real_part = jnp.real(impedance)
        assert jnp.all(real_part >= 0), "System is not passive"


# ============================================================================
# Test Data Generation
# ============================================================================

@pytest.fixture
def synthetic_loudspeaker_data(rng_key):
    """Generate synthetic loudspeaker measurement data."""
    # Parameters for realistic loudspeaker
    params = {
        'Re': 6.8,      # Electrical resistance [Ω]
        'Le': 0.5e-3,   # Electrical inductance [H]
        'Bl': 3.2,      # Force factor [N/A]
        'M': 12e-3,     # Moving mass [kg]
        'K': 1200,      # Stiffness [N/m]
        'Rm': 0.8,      # Mechanical resistance [N·s/m]
    }
    
    # Generate time series
    sample_rate = 48000  # Hz
    duration = 1.0       # seconds
    n_samples = int(sample_rate * duration)
    t = jnp.linspace(0, duration, n_samples)
    
    # Generate pink noise input
    rng_key, subkey = jax.random.split(rng_key)
    white_noise = jax.random.normal(subkey, (n_samples,))
    
    # Simple pink noise filter (1/f)
    freqs = jnp.fft.fftfreq(n_samples, 1/sample_rate)
    pink_filter = 1 / jnp.sqrt(jnp.abs(freqs) + 1e-10)
    pink_filter = pink_filter.at[0].set(0)  # Remove DC
    
    pink_noise = jnp.real(jnp.fft.ifft(jnp.fft.fft(white_noise) * pink_filter))
    voltage = 2.0 * pink_noise  # 2V RMS
    
    # Simple linear loudspeaker model for testing
    # This is a simplified model - real implementation will be more complex
    current = voltage / params['Re']  # Simplified electrical model
    displacement = jnp.zeros_like(voltage)  # Placeholder
    velocity = jnp.zeros_like(voltage)  # Placeholder
    
    return {
        'voltage': voltage,
        'current': current,
        'displacement': displacement,
        'velocity': velocity,
        'sample_rate': sample_rate,
        'time': t,
        'parameters': params
    }


@pytest.fixture
def loudspeaker_parameters():
    """Standard loudspeaker parameters for testing."""
    return {
        'Re': 6.8,      # Electrical resistance [Ω]
        'Le': 0.5e-3,   # Electrical inductance [H]
        'Bl': 3.2,      # Force factor [N/A]
        'M': 12e-3,     # Moving mass [kg]
        'K': 1200,      # Stiffness [N/m]
        'Rm': 0.8,      # Mechanical resistance [N·s/m]
        # Nonlinear parameters
        'Bl_nl': [0.0, -0.1, 0.0, 0.0],  # Bl(x) polynomial coefficients
        'K_nl': [0.0, 0.0, 100.0, 0.0],  # K(x) polynomial coefficients
        'L_nl': [0.0, 0.0, 0.0, 0.0],    # L(x) polynomial coefficients
    }


# ============================================================================
# Property-Based Testing Strategies
# ============================================================================

@st.composite
def loudspeaker_parameter_strategy(draw):
    """Strategy for generating valid loudspeaker parameters."""
    return {
        'Re': draw(st.floats(min_value=1.0, max_value=20.0)),
        'Le': draw(st.floats(min_value=0.1e-3, max_value=2.0e-3)),
        'Bl': draw(st.floats(min_value=1.0, max_value=10.0)),
        'M': draw(st.floats(min_value=5e-3, max_value=50e-3)),
        'K': draw(st.floats(min_value=500, max_value=5000)),
        'Rm': draw(st.floats(min_value=0.1, max_value=5.0)),
    }


@st.composite
def signal_strategy(draw, n_samples=1000):
    """Strategy for generating test signals."""
    signal_type = draw(st.sampled_from(['sine', 'chirp', 'noise', 'impulse']))
    
    if signal_type == 'sine':
        freq = draw(st.floats(min_value=10, max_value=1000))
        amplitude = draw(st.floats(min_value=0.1, max_value=10.0))
        t = jnp.linspace(0, 1, n_samples)
        return amplitude * jnp.sin(2 * jnp.pi * freq * t)
    
    elif signal_type == 'chirp':
        f0 = draw(st.floats(min_value=10, max_value=100))
        f1 = draw(st.floats(min_value=100, max_value=1000))
        amplitude = draw(st.floats(min_value=0.1, max_value=10.0))
        t = jnp.linspace(0, 1, n_samples)
        return amplitude * jnp.sin(2 * jnp.pi * (f0 + (f1 - f0) * t) * t)
    
    elif signal_type == 'noise':
        amplitude = draw(st.floats(min_value=0.1, max_value=10.0))
        return amplitude * jax.random.normal(jax.random.PRNGKey(42), (n_samples,))
    
    else:  # impulse
        amplitude = draw(st.floats(min_value=0.1, max_value=10.0))
        signal = jnp.zeros(n_samples)
        return signal.at[n_samples//2].set(amplitude)


# ============================================================================
# Performance Testing Utilities
# ============================================================================

@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        'min_time': 0.1,  # Minimum time per benchmark
        'max_time': 10.0,  # Maximum time per benchmark
        'warmup': True,    # Enable warmup runs
        'rounds': 5,       # Number of benchmark rounds
    }


class PerformanceTestMixin:
    """Mixin class for performance testing utilities."""
    
    @staticmethod
    def assert_execution_time(func: Callable, max_time: float, *args, **kwargs) -> None:
        """Assert function executes within specified time."""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        assert execution_time < max_time, f"Function took {execution_time:.3f}s, expected < {max_time}s"
        return result
    
    @staticmethod
    def assert_memory_usage(func: Callable, max_memory_mb: float, *args, **kwargs) -> None:
        """Assert function uses less than specified memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        assert memory_used < max_memory_mb, f"Function used {memory_used:.1f}MB, expected < {max_memory_mb}MB"
        return result


# ============================================================================
# Mock Objects and Fixtures
# ============================================================================

@pytest.fixture
def mock_mat_file():
    """Mock .mat file for testing data loading."""
    mock_data = {
        'voltage': np.random.randn(1000),
        'current': np.random.randn(1000),
        'displacement': np.random.randn(1000),
        'velocity': np.random.randn(1000),
        'sample_rate': np.array([[48000.0]])
    }
    return mock_data


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_optimizer():
    """Mock optimizer for testing."""
    mock_opt = Mock()
    mock_opt.step.return_value = {'loss': 0.1, 'grad_norm': 0.01}
    mock_opt.state = {'iteration': 0}
    return mock_opt


# ============================================================================
# Test Decorators and Utilities
# ============================================================================

def requires_gpu(func):
    """Decorator to skip tests if GPU is not available."""
    return pytest.mark.skipif(
        not jax.devices('gpu'),
        reason="GPU not available"
    )(func)


def slow_test(func):
    """Decorator for slow tests."""
    return pytest.mark.slow(func)


def mathematical_test(func):
    """Decorator for mathematical correctness tests."""
    return pytest.mark.mathematical(func)


def physical_test(func):
    """Decorator for physical constraint tests."""
    return pytest.mark.physical(func)


def regression_test(func):
    """Decorator for regression tests."""
    return pytest.mark.regression(func)


# ============================================================================
# Hypothesis Settings
# ============================================================================

# Global hypothesis settings for faster test execution
settings.register_profile("fast", max_examples=10, deadline=1000)
settings.register_profile("thorough", max_examples=100, deadline=5000)
settings.load_profile("fast")


# ============================================================================
# Test Collection Hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test names
        if "test_mathematical" in item.name:
            item.add_marker(pytest.mark.mathematical)
        if "test_physical" in item.name:
            item.add_marker(pytest.mark.physical)
        if "test_performance" in item.name:
            item.add_marker(pytest.mark.performance)
        if "test_slow" in item.name:
            item.add_marker(pytest.mark.slow)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "mathematical: Mathematical correctness tests")
    config.addinivalue_line("markers", "physical: Physical constraint tests")
