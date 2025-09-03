# 🧪 Comprehensive Testing Framework Guide

## Overview

This guide explains how to use the comprehensive testing framework for loudspeaker system identification. The framework provides standardized testing of different system identification methods with the exact metrics you requested:

- **Model parameters**: Fitted parameters from each method
- **Error timeseries**: Time-domain error between true and predicted outputs
- **Final loss**: Mean squared error loss
- **Final R²**: Coefficient of determination

## 🎯 Key Features

### Ground Truth Model
- **Based on Heuchel et al. ICA 2022**: Implements the reference methodology from [quant-comp-ls-mod-ica22](https://github.com/fhchl/quant-comp-ls-mod-ica22)
- **Nonlinear Dynamics**: Full nonlinear loudspeaker model with Bl(x), K(x), L(x,i) nonlinearities
- **Eddy Current Model**: L2R2 model for high-frequency behavior
- **Synthetic Data Generation**: Realistic measurement data with configurable noise

### Comprehensive Testing
- **Multiple Methods**: Test different system identification approaches
- **Standardized Metrics**: Consistent evaluation across all methods
- **Statistical Analysis**: R², NRMSE, MAE, correlation coefficients
- **Performance Benchmarking**: Training time and convergence analysis

### Dynax Integration
- **Reference Implementation**: Follows Heuchel et al. methodology exactly
- **CSD Matching**: Cross-spectral density matching for linear parameters
- **ML Estimation**: Maximum likelihood estimation for nonlinear parameters
- **Forward Model**: Complete ODE integration with Diffrax

## 🚀 Quick Start

### 1. Run Verification Tests
```bash
python run_tests.py
```

This will verify that all components work correctly.

### 2. Run Comprehensive Tests
```bash
# Basic test with standard ground truth
python src/run_comprehensive_tests.py

# Test with different ground truth models
python src/run_comprehensive_tests.py --ground-truth highly_nonlinear
python src/run_comprehensive_tests.py --ground-truth linear

# Test with different excitations
python src/run_comprehensive_tests.py --excitation sine
python src/run_comprehensive_tests.py --excitation chirp
python src/run_comprehensive_tests.py --excitation multitone

# Test with different noise levels
python src/run_comprehensive_tests.py --noise-level 0.05
python src/run_comprehensive_tests.py --noise-level 0.001
```

### 3. View Results
The framework generates comprehensive reports in the `test_results/` directory:

- **`comprehensive_test_report.json`**: Detailed results in JSON format
- **`test_summary.txt`**: Human-readable summary
- **`error_timeseries.png`**: Error plots for all methods
- **`parameter_comparison.png`**: Parameter comparison with ground truth
- **`performance_comparison.png`**: R² and training time comparison

## 📊 Output Metrics

### Required Metrics (As Requested)

#### 1. Model Parameters
```json
{
  "model_parameters": {
    "Re": 6.8,
    "Le": 0.0005,
    "Bl": 3.2,
    "M": 0.012,
    "K": 1200.0,
    "Rm": 0.8,
    "Bl_nl": [0.0, 0.0, -50.0, -0.1],
    "K_nl": [0.0, 0.0, 100.0, 0.0]
  }
}
```

#### 2. Error Timeseries
```json
{
  "error_timeseries": {
    "current": [0.001, -0.002, 0.003, ...],
    "velocity": [0.0001, -0.0002, 0.0003, ...]
  }
}
```

#### 3. Final Loss
```json
{
  "final_loss": 0.001234
}
```

#### 4. Final R²
```json
{
  "final_r2": 0.9876
}
```

### Additional Metrics

#### Performance Metrics
- **Training Time**: Execution time in seconds
- **Convergence Info**: Optimization convergence details
- **NRMSE**: Normalized root mean square error
- **MAE**: Mean absolute error
- **Correlation**: Pearson correlation coefficient

#### Statistical Analysis
- **Method Ranking**: Ranked by R² score
- **Best Method**: Highest performing method
- **Performance Summary**: Comparative statistics

## 🔬 Ground Truth Models

### Standard Model
```python
from src.ground_truth_model import create_standard_ground_truth

model = create_standard_ground_truth()
# Typical nonlinear loudspeaker with moderate nonlinearities
```

### Highly Nonlinear Model
```python
from src.ground_truth_model import create_highly_nonlinear_ground_truth

model = create_highly_nonlinear_ground_truth()
# Strong nonlinearities for challenging identification
```

### Linear Model
```python
from src.ground_truth_model import create_linear_ground_truth

model = create_linear_ground_truth()
# No nonlinearities for baseline comparison
```

## 🧪 Test Data Generation

### Excitation Types

#### Pink Noise
```python
test_data = tester.generate_test_data(
    excitation_type='pink_noise',
    duration=1.0,
    amplitude=2.0,
    noise_level=0.01
)
```

#### Sine Wave
```python
test_data = tester.generate_test_data(
    excitation_type='sine',
    duration=1.0,
    amplitude=1.0,
    noise_level=0.01
)
```

#### Chirp Signal
```python
test_data = tester.generate_test_data(
    excitation_type='chirp',
    duration=1.0,
    amplitude=1.0,
    noise_level=0.01
)
```

#### Multitone
```python
test_data = tester.generate_test_data(
    excitation_type='multitone',
    duration=1.0,
    amplitude=1.0,
    noise_level=0.01
)
```

## 🔧 Adding New Methods

### 1. Create Method Function
```python
def my_identification_method(u: jnp.ndarray, y: jnp.ndarray, **kwargs) -> Dict[str, Any]:
    """
    Your system identification method.
    
    Args:
        u: Input voltage [V]
        y: Output measurements [current, velocity]
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with:
        - 'model': Fitted model object
        - 'parameters': Model parameters dict
        - 'predictions': Predicted outputs
        - 'convergence': Convergence info
    """
    # Your implementation here
    
    return {
        'model': fitted_model,
        'parameters': parameters,
        'predictions': predictions,
        'convergence': {'converged': True, 'iterations': 100}
    }
```

### 2. Add to Test Suite
```python
# In src/run_comprehensive_tests.py
methods = {
    'My Method': my_identification_method,
    # ... other methods
}
```

### 3. Run Tests
```bash
python src/run_comprehensive_tests.py
```

## 📈 Understanding Results

### R² Score Interpretation
- **R² = 1.0**: Perfect prediction
- **R² > 0.9**: Excellent fit
- **R² > 0.8**: Good fit
- **R² > 0.7**: Acceptable fit
- **R² < 0.7**: Poor fit

### Loss Interpretation
- **Lower is better**: MSE loss should be minimized
- **Compare relative values**: Look at relative differences between methods
- **Consider noise level**: Loss should be above noise floor

### Parameter Comparison
- **Close to ground truth**: Good parameter estimation
- **Systematic bias**: Method-specific estimation errors
- **High variance**: Unstable estimation

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Make sure JAX is installed
pip install jax jaxlib

# For GPU support
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### Memory Issues
```python
# Reduce test duration
python src/run_comprehensive_tests.py --duration 0.1

# Reduce sample rate
test_data = tester.generate_test_data(sample_rate=24000)
```

#### Convergence Issues
```python
# Check convergence info in results
result = tester.test_method(...)
print(result.convergence_info)
```

### Debug Mode
```python
# Enable verbose output
python src/run_comprehensive_tests.py --verbose

# Check individual components
python run_tests.py
```

## 📚 References

1. **Heuchel, F. M., & Agerkvist, F. T. (2022)**: "A quantitative comparison of linear and nonlinear loudspeaker models", ICA 2022
2. **Reference Implementation**: [quant-comp-ls-mod-ica22](https://github.com/fhchl/quant-comp-ls-mod-ica22)
3. **Dynax Documentation**: [Dynax GitHub](https://github.com/fhchl/dynax)
4. **JAX Documentation**: [JAX Documentation](https://jax.readthedocs.io/)

## 🤝 Contributing

To add new system identification methods:

1. **Follow the interface**: Implement the required return format
2. **Add tests**: Include unit tests for your method
3. **Document parameters**: Clearly document all parameters
4. **Validate results**: Ensure your method produces reasonable results
5. **Update documentation**: Add your method to this guide

## 📞 Support

For questions or issues:

1. **Check the logs**: Look at console output for error messages
2. **Run verification**: Use `python run_tests.py` to check components
3. **Review examples**: Look at existing method implementations
4. **Check documentation**: Refer to this guide and code comments

---

**Happy Testing!** 🎵✨
