# Test-Driven Development Methodology for Loudspeaker System Identification

## Abstract

This document outlines the comprehensive Test-Driven Development (TDD) methodology employed in the loudspeaker-jax project. Our TDD approach ensures mathematical correctness, physical validity, and robust implementation of complex system identification algorithms.

## 1. TDD Philosophy and Principles

### 1.1 Core TDD Cycle

Our TDD implementation follows the classic Red-Green-Refactor cycle:

1. **Red**: Write a failing test that defines the desired behavior
2. **Green**: Write the minimal code to make the test pass
3. **Refactor**: Improve the code while keeping all tests green
4. **Repeat**: Continue the cycle for new features

### 1.2 Scientific TDD Principles

For scientific computing, we extend traditional TDD with:

- **Mathematical Correctness**: Tests verify mathematical properties and invariants
- **Physical Validity**: Tests ensure physical constraints are satisfied
- **Numerical Stability**: Tests check for numerical robustness
- **Performance Regression**: Tests detect performance degradations
- **Uncertainty Quantification**: Tests validate statistical properties

## 2. Test Categories and Strategies

### 2.1 Unit Tests

**Purpose**: Test individual functions and classes in isolation.

**Examples**:
```python
def test_force_factor_calculation(self):
    """Test Bl(x) force factor calculation."""
    model = LoudspeakerModel(Bl=3.2, Bl_nl=[0.0, -0.1, 0.0, 0.0])
    
    # Test at x=0
    x_zero = jnp.array(0.0)
    Bl_zero = model.force_factor(x_zero)
    assert jnp.isclose(Bl_zero, 3.2)
    
    # Test at x=0.001 (1mm)
    x_test = jnp.array(0.001)
    Bl_test = model.force_factor(x_test)
    expected = 3.2 - 0.1 * 0.001
    assert jnp.isclose(Bl_test, expected)
```

**Coverage**: All public methods, edge cases, error conditions.

### 2.2 Integration Tests

**Purpose**: Test complete workflows and component interactions.

**Examples**:
```python
def test_diffrax_integration(self):
    """Test integration with Diffrax ODE solver."""
    model = LoudspeakerModel(Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
    
    def ode_func(t, y, args):
        u = args
        return model.dynamics(y, u)
    
    solution = dfx.diffeqsolve(
        dfx.ODETerm(ode_func),
        dfx.Tsit5(),
        0.0, 1.0, dt0=0.01,
        y0=jnp.zeros(4),
        args=jnp.array(2.0)
    )
    
    assert solution.ys.shape[0] > 0
    assert jnp.all(jnp.isfinite(solution.ys))
```

**Coverage**: End-to-end workflows, external library integration, data pipelines.

### 2.3 Property-Based Tests

**Purpose**: Test mathematical invariants and properties using Hypothesis.

**Examples**:
```python
@given(loudspeaker_parameter_strategy())
def test_parameter_physical_constraints(self, params):
    """Test that all parameters satisfy physical constraints."""
    model = LoudspeakerModel(**params)
    
    # All parameters should be positive
    self.assert_positive(jnp.array([model.Re, model.Le, model.Bl, model.M, model.K, model.Rm]))
    
    # Mass should be reasonable (between 1g and 100g)
    assert 1e-3 < model.M < 100e-3
    
    # Stiffness should be reasonable (between 100 and 10000 N/m)
    assert 100 < model.K < 10000
```

**Coverage**: Mathematical invariants, physical constraints, edge cases.

### 2.4 Performance Tests

**Purpose**: Benchmark performance and detect regressions.

**Examples**:
```python
def test_optimization_performance(self):
    """Test that optimization completes within reasonable time."""
    optimizer = GaussNewtonOptimizer(max_iterations=100)
    
    # Test execution time
    result = self.assert_execution_time(
        optimizer.optimize, 
        max_time=10.0,  # 10 seconds
        initial_model, u, y
    )
    
    # Test memory usage
    self.assert_memory_usage(
        optimizer.optimize,
        max_memory_mb=100,  # 100 MB
        initial_model, u, y
    )
```

**Coverage**: Algorithm performance, memory usage, scalability.

### 2.5 Mathematical Tests

**Purpose**: Verify mathematical correctness and numerical stability.

**Examples**:
```python
def test_jacobian_calculation(self):
    """Test that Jacobians can be calculated correctly."""
    model = LoudspeakerModel(Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8)
    
    x = jnp.array([0.1, 0.001, 0.01, 0.05])
    u = jnp.array(2.0)
    
    # Test state Jacobian
    df_dx = jax.jacfwd(model.dynamics, argnums=0)(x, u)
    assert df_dx.shape == (4, 4)
    assert jnp.all(jnp.isfinite(df_dx))
    
    # Test input Jacobian
    df_du = jax.jacfwd(model.dynamics, argnums=1)(x, u)
    assert df_du.shape == (4,)
    assert jnp.all(jnp.isfinite(df_du))
```

**Coverage**: Gradient calculations, matrix properties, numerical stability.

### 2.6 Physical Tests

**Purpose**: Validate physical constraints and conservation laws.

**Examples**:
```python
def test_energy_conservation_linear_case(self):
    """Test energy conservation for linear case."""
    model = LoudspeakerModel(
        Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8,
        Bl_nl=[0.0, 0.0, 0.0, 0.0],  # No nonlinearities
        K_nl=[0.0, 0.0, 0.0, 0.0],
        L_nl=[0.0, 0.0, 0.0, 0.0]
    )
    
    # Test with zero input (free oscillation)
    x = jnp.array([0.0, 0.001, 0.01, 0.0])
    u = jnp.array(0.0)
    
    dxdt = model.dynamics(x, u)
    
    # For linear case with no input, energy should be conserved
    # (ignoring damping for this test)
    self.assert_energy_conservation(initial_energy, final_energy)
```

**Coverage**: Energy conservation, causality, passivity, physical bounds.

## 3. Test Infrastructure

### 3.1 Test Configuration

**pytest.ini**:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

addopts = 
    --verbose
    --tb=short
    --strict-markers
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=90

markers =
    unit: Unit tests for individual functions
    integration: Integration tests for complete workflows
    property: Property-based tests using Hypothesis
    performance: Performance and benchmark tests
    mathematical: Tests for mathematical correctness
    physical: Tests for physical constraints and invariants
```

### 3.2 Test Utilities

**MathematicalTestMixin**:
```python
class MathematicalTestMixin:
    @staticmethod
    def assert_close(a: jnp.ndarray, b: jnp.ndarray, 
                    rtol: float = 1e-6, atol: float = 1e-8) -> None:
        """Assert two arrays are close with specified tolerances."""
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    
    @staticmethod
    def assert_positive_definite(matrix: jnp.ndarray) -> None:
        """Assert matrix is positive definite."""
        eigenvals = jnp.linalg.eigvals(matrix)
        assert jnp.all(eigenvals > 0)
```

**PhysicalConstraintMixin**:
```python
class PhysicalConstraintMixin:
    @staticmethod
    def assert_energy_conservation(initial_energy: float, final_energy: float, 
                                 tolerance: float = 1e-3) -> None:
        """Assert energy conservation within tolerance."""
        energy_change = abs(final_energy - initial_energy) / abs(initial_energy)
        assert energy_change < tolerance
    
    @staticmethod
    def assert_causality(impulse_response: jnp.ndarray, 
                        sample_rate: float) -> None:
        """Assert system is causal (no response before t=0)."""
        assert jnp.all(impulse_response[:int(sample_rate * 0.001)] == 0)
```

### 3.3 Test Data Generation

**Synthetic Data Fixtures**:
```python
@pytest.fixture
def synthetic_loudspeaker_data(rng_key):
    """Generate synthetic loudspeaker measurement data."""
    params = {
        'Re': 6.8, 'Le': 0.5e-3, 'Bl': 3.2,
        'M': 12e-3, 'K': 1200, 'Rm': 0.8
    }
    
    sample_rate = 48000
    duration = 1.0
    n_samples = int(sample_rate * duration)
    t = jnp.linspace(0, duration, n_samples)
    
    # Generate pink noise input
    rng_key, subkey = jax.random.split(rng_key)
    white_noise = jax.random.normal(subkey, (n_samples,))
    
    # Simple pink noise filter
    freqs = jnp.fft.fftfreq(n_samples, 1/sample_rate)
    pink_filter = 1 / jnp.sqrt(jnp.abs(freqs) + 1e-10)
    pink_filter = pink_filter.at[0].set(0)
    
    pink_noise = jnp.real(jnp.fft.ifft(jnp.fft.fft(white_noise) * pink_filter))
    voltage = 2.0 * pink_noise
    
    return {
        'voltage': voltage,
        'current': voltage / params['Re'],
        'displacement': jnp.zeros_like(voltage),
        'velocity': jnp.zeros_like(voltage),
        'sample_rate': sample_rate,
        'time': t,
        'parameters': params
    }
```

**Property-Based Strategies**:
```python
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
```

## 4. TDD Workflow Implementation

### 4.1 Development Cycle

1. **Write Failing Test**: Define expected behavior
2. **Run Test**: Confirm it fails (Red)
3. **Implement Code**: Minimal implementation to pass
4. **Run Test**: Confirm it passes (Green)
5. **Refactor**: Improve code quality
6. **Run All Tests**: Ensure no regressions

### 4.2 Test-Driven Implementation Example

**Step 1: Write Failing Test**
```python
def test_force_factor_calculation(self):
    """Test Bl(x) force factor calculation."""
    model = LoudspeakerModel(Bl=3.2, Bl_nl=[0.0, -0.1, 0.0, 0.0])
    
    x_test = jnp.array(0.001)
    Bl_test = model.force_factor(x_test)
    expected = 3.2 - 0.1 * 0.001
    assert jnp.isclose(Bl_test, expected)
```

**Step 2: Implement Minimal Code**
```python
def force_factor(self, x: jnp.ndarray) -> jnp.ndarray:
    """Calculate position-dependent force factor Bl(x)."""
    return self.Bl + jnp.polyval(jnp.append(self.Bl_nl, 0), x)
```

**Step 3: Refactor and Add More Tests**
```python
def test_force_factor_edge_cases(self):
    """Test force factor at edge cases."""
    model = LoudspeakerModel(Bl=3.2, Bl_nl=[0.0, -0.1, 0.0, 0.0])
    
    # Test at x=0
    assert jnp.isclose(model.force_factor(jnp.array(0.0)), 3.2)
    
    # Test with large displacement
    x_large = jnp.array(0.01)  # 10mm
    Bl_large = model.force_factor(x_large)
    assert Bl_large > 0  # Should remain positive
```

### 4.3 Continuous Integration

**GitHub Actions Workflow**:
```yaml
name: TDD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install pixi
        pixi install
    
    - name: Run unit tests
      run: pixi run python tests/run_tests.py --test-type unit
    
    - name: Run mathematical tests
      run: pixi run python tests/run_tests.py --test-type mathematical
    
    - name: Run performance tests
      run: pixi run python tests/run_tests.py --test-type performance
    
    - name: Check coverage
      run: pixi run python tests/run_tests.py --coverage-check 90
```

## 5. Quality Assurance

### 5.1 Code Coverage

- **Minimum Coverage**: 90% line coverage required
- **Branch Coverage**: 85% branch coverage required
- **Critical Paths**: 100% coverage for mathematical functions

### 5.2 Performance Benchmarks

- **Regression Detection**: 10% performance degradation threshold
- **Memory Usage**: Maximum memory usage limits
- **Scalability**: Performance tests with varying data sizes

### 5.3 Mathematical Validation

- **Numerical Stability**: Tests for numerical robustness
- **Gradient Accuracy**: Finite difference validation
- **Matrix Properties**: Positive definiteness, symmetry, orthogonality

### 5.4 Physical Validation

- **Energy Conservation**: Tests for energy conservation
- **Causality**: Tests for causal system behavior
- **Passivity**: Tests for passive system properties
- **Parameter Bounds**: Tests for physically reasonable parameters

## 6. Best Practices

### 6.1 Test Design

1. **Single Responsibility**: Each test should verify one specific behavior
2. **Clear Naming**: Test names should describe the expected behavior
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Independent Tests**: Tests should not depend on each other
5. **Deterministic**: Tests should produce consistent results

### 6.2 Test Maintenance

1. **Regular Updates**: Update tests when requirements change
2. **Refactoring**: Refactor tests along with code
3. **Documentation**: Document complex test logic
4. **Performance**: Keep tests fast and efficient
5. **Coverage**: Maintain high test coverage

### 6.3 Scientific Computing Specifics

1. **Tolerance Testing**: Use appropriate numerical tolerances
2. **Property Testing**: Test mathematical invariants
3. **Physical Constraints**: Validate physical laws
4. **Edge Cases**: Test boundary conditions and extreme values
5. **Random Testing**: Use property-based testing for robustness

## 7. Tools and Libraries

### 7.1 Testing Framework

- **pytest**: Primary testing framework
- **hypothesis**: Property-based testing
- **pytest-cov**: Coverage reporting
- **pytest-benchmark**: Performance testing
- **pytest-xdist**: Parallel test execution

### 7.2 JAX-Specific Testing

- **jax.test_utils**: JAX testing utilities
- **jax.numpy.testing**: NumPy-compatible assertions
- **jax.random**: Reproducible random number generation

### 7.3 Scientific Testing

- **scipy.stats**: Statistical testing
- **numpy.testing**: Numerical testing utilities
- **matplotlib**: Test visualization (non-interactive)

## 8. Conclusion

Our TDD methodology ensures that the loudspeaker system identification implementation is:

- **Mathematically Correct**: All algorithms satisfy mathematical properties
- **Physically Valid**: All models respect physical constraints
- **Numerically Stable**: All computations are robust
- **Well-Tested**: Comprehensive test coverage
- **Maintainable**: Clear, documented, and refactored code

This rigorous approach is essential for scientific computing applications where correctness and reliability are paramount.
