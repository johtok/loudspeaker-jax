# 🎵 Loudspeaker-JAX: Advanced System Identification & Approximation

A comprehensive research framework for loudspeaker system identification and approximation using the cutting-edge JAX ecosystem. This project implements state-of-the-art methods for modeling loudspeaker dynamics, from classical linear system identification to modern Bayesian inference and Gaussian Process surrogates.

## 🔬 Research Objectives

This project addresses two fundamental challenges in loudspeaker modeling:

1. **System Identification**: Determine the physical parameters of loudspeaker systems from measurement data
2. **System Approximation**: Develop accurate surrogate models for real-time applications

### Key Research Questions

- How can we leverage modern JAX-based optimization for superior loudspeaker parameter estimation?
- What is the role of uncertainty quantification in loudspeaker modeling?
- How do different nonlinear modeling approaches compare in terms of accuracy and computational efficiency?
- Can Gaussian Process surrogates capture unmodeled nonlinearities better than polynomial expansions?

## 🚀 Methodology Framework

### 1. Grey-Box Physical Modeling
- **Diffrax** + **JAXopt**: Differentiable ODE solving with Gauss-Newton/Levenberg-Marquardt optimization
- Comprehensive loudspeaker model with nonlinearities: Bl(x), K(x), L(x,i), eddy currents, thermal effects
- Multi-scale optimization: linear → nonlinear → Bayesian

### 2. Probabilistic State-Space Modeling
- **Dynamax**: Nonlinear-Gaussian state-space models for joint state/parameter learning
- Latent state estimation with principled uncertainty quantification
- Sensor fusion for current, voltage, displacement, and velocity measurements

### 3. Bayesian Parameter Inference
- **NumPyro** + **BlackJAX**: Probabilistic programming with fast HMC/NUTS sampling
- Parameter correlation analysis and uncertainty propagation
- MAP estimation with full posterior distributions

### 4. Gaussian Process Surrogates
- **GPJax**: GP regression for unmodeled nonlinearities and thermal drift
- Hybrid physics-informed models combining analytical and data-driven components
- Fast surrogate models for real-time applications

### 5. Classical System Identification
- **jax-sysid**: Subspace identification and nonlinear system identification
- Baseline methods for comparison and initialization
- L1/group-Lasso regularization for sparse models

## 📊 Dataset: DTU Loudspeaker Measurements

The project uses the comprehensive DTU loudspeaker dataset featuring:

- **Pink noise recordings** (5Hz-2000Hz, 8Vrms) for broadband system identification
- **Multiple measurement protocols** (LPM, LSI-4, LSI-8) for validation across excitation levels
- **Physical measurements**: Bl(x), K(x), L(x,i) curves for ground truth validation
- **Thermal measurements**: Parameter drift analysis under different operating conditions

## 🛠️ Technology Stack

### Core JAX Ecosystem
* **JAX**: High-performance machine learning and scientific computing
* **Diffrax**: Differentiable differential equation solving
* **Equinox**: Neural network library with PyTorch-like API
* **JAXopt**: Optimization algorithms (Gauss-Newton, Levenberg-Marquardt, L-BFGS)

### Probabilistic Programming
* **NumPyro**: Probabilistic programming with NumPy backend
* **BlackJAX**: Fast MCMC sampling (HMC, NUTS, VI)

### Advanced Modeling
* **GPJax**: Gaussian processes in JAX
* **Dynamax**: Probabilistic state-space models
* **jax-sysid**: System identification algorithms

### Project Management
* **Hydra**: Configuration management
* **Pixi**: Dependency management and environment handling
* **pdoc**: Automatic API documentation generation
* **pre-commit**: Code quality automation

## Project Structure

```bash
.
├── config
│   ├── main.yaml                   # Main configuration file
│   ├── model                       # Configurations for training model
│   │   ├── model1.yaml             # First variation of parameters to train model
│   │   └── model2.yaml             # Second variation of parameters to train model
│   └── process                     # Configurations for processing data
│       ├── process1.yaml           # First variation of parameters to process data
│       └── process2.yaml           # Second variation of parameters to process data
├── data
│   ├── final                       # data after training the model
│   ├── processed                   # data after processing
│   └── raw                         # raw data
├── docs                            # documentation for your project
├── .gitignore                      # ignore files that cannot commit to Git
├── models                          # store models
├── notebooks                       # store notebooks
├── .pre-commit-config.yaml         # configurations for pre-commit
├── pyproject.toml                  # dependencies for pixi
├── README.md                       # describe your project
├── src                             # store source code
│   ├── __init__.py                 # make src a Python module
│   ├── process.py                  # process data before training model
│   ├── train_model.py              # train model
│   └── utils.py                    # store helper functions
└── tests                           # store tests
    ├── __init__.py                 # make tests a Python module
    ├── test_process.py             # test functions for process.py
    └── test_train_model.py         # test functions for train_model.py
```

## Version Control Setup

1. Initialize Git in your project directory:
```bash
git init
```

2. Add your remote repository:
```bash
# For HTTPS
git remote add origin https://github.com/username/repository-name.git

# For SSH
git remote add origin git@github.com:username/repository-name.git
```

3. Create and switch to a new branch:
```bash
git checkout -b main
```

4. Add and commit your files:
```bash
git add .
git commit -m "Initial commit"
```

5. Push to your remote repository:
```bash
git push -u origin main
```

## Set up the environment
1. Install [Pixi](https://pixi.sh)

2. Activate the virtual environment:

```bash
pixi shell
```

3. activation of other environments:

- To check available environment:

```bash
pixi status
```

- To run an environment run:
```bash
pixi s -e ENVIRONTMENT
```

4. Run Python scripts:

```bash
# Run directly with pixi
pixi run python src/process.py

# Or after activating the virtual environment
python src/process.py
```

## 🎯 **ROCK-SOLID IMPLEMENTATION PLAN**

### 📋 **Phase-Based Development Strategy**

This project follows a comprehensive 5-phase development plan ensuring mathematical rigor and scientific excellence:

#### **Phase 1: Foundation (Weeks 1-3)** 🔄 *In Progress*
- **Environment Setup**: JAX ecosystem configuration and verification
- **Data Infrastructure**: DTU dataset loading and validation
- **Basic Model**: Core LoudspeakerModel with linear parameters
- **Linear Identification**: CSD matching and subspace methods
- **Validation**: Comprehensive testing and validation

#### **Phase 2: Nonlinear Methods (Weeks 4-6)** ⏳ *Pending*
- **Nonlinear Extensions**: Bl(x), K(x), L(x,i) nonlinearities
- **ODE Integration**: Diffrax-based differential equation solving
- **Advanced Optimization**: Gauss-Newton and Levenberg-Marquardt
- **Multi-Scale Strategy**: Hierarchical optimization approach

#### **Phase 3: Advanced Methods (Weeks 7-9)** ⏳ *Pending*
- **Bayesian Inference**: NumPyro + BlackJAX for uncertainty quantification
- **State-Space Modeling**: Dynamax for probabilistic state estimation
- **GP Surrogates**: GPJax for unmodeled nonlinearities
- **Hybrid Modeling**: Physics-informed machine learning

#### **Phase 4: Analysis & Validation (Weeks 10-11)** ⏳ *Pending*
- **Comparative Analysis**: Rigorous method comparison
- **Performance Benchmarking**: Comprehensive performance analysis
- **Statistical Validation**: Significance testing and confidence intervals
- **Documentation**: Complete API and research documentation

#### **Phase 5: Publication & Release (Weeks 12-13)** ⏳ *Pending*
- **Research Publication**: Academic paper preparation
- **Open-Source Release**: PyPI package and community resources
- **Final Validation**: Independent validation and peer review

### 🧪 **Test-Driven Development (TDD) Framework**

This project follows rigorous Test-Driven Development principles with comprehensive testing infrastructure:

#### **Test Categories**
- **Unit Tests**: Individual function and class testing
- **Integration Tests**: Complete workflow testing
- **Property-Based Tests**: Mathematical invariant testing using Hypothesis
- **Performance Tests**: Benchmarking and regression testing
- **Mathematical Tests**: Correctness of mathematical formulations
- **Physical Tests**: Validation of physical constraints and laws

#### **Running Tests**
```bash
# Run all tests with coverage
pixi run python tests/run_tests.py --test-type all

# Run specific test categories
pixi run python tests/run_tests.py --test-type unit
pixi run python tests/run_tests.py --test-type mathematical
pixi run python tests/run_tests.py --test-type performance

# Run TDD cycle for specific module
pixi run python tests/run_tests.py --tdd-file test_loudspeaker_model.py

# Generate comprehensive test report
pixi run python tests/run_tests.py --report

# Check test coverage (minimum 90%)
pixi run python tests/run_tests.py --coverage-check 90
```

#### **TDD Workflow**
1. **Red**: Write failing tests first
2. **Green**: Implement minimal code to pass tests
3. **Refactor**: Improve code while keeping tests green
4. **Repeat**: Continue cycle for new features

#### **Test Configuration**
- **Coverage**: Minimum 90% code coverage required
- **Performance**: Benchmark regression detection
- **Mathematical**: Property-based testing with Hypothesis
- **Physical**: Constraint validation and energy conservation

### 📊 **Current Project Status**

**Overall Progress**: 15% Complete

**Current Phase**: Phase 1 - Foundation (In Progress)

**Completed**:
- ✅ Comprehensive TDD framework
- ✅ Mathematical foundations documentation
- ✅ Project architecture and structure
- ✅ JAX ecosystem configuration

**In Progress**:
- 🔄 Environment verification and testing
- 🔄 DTU dataset loading infrastructure

**Next Milestones**:
- Basic LoudspeakerModel implementation
- Linear system identification methods
- Phase 1 validation and testing

### 📚 **Documentation**

- **[Implementation Plan](docs/implementation_plan.md)**: Comprehensive project roadmap
- **[Technical Roadmap](docs/technical_roadmap.md)**: Step-by-step implementation guide
- **[Mathematical Foundations](docs/mathematical_foundations.md)**: Complete mathematical derivations
- **[TDD Methodology](docs/tdd_methodology.md)**: Test-driven development approach
- **[Project Status](docs/project_status.md)**: Current progress and status dashboard

## Set up pre-commit hooks
Set up pre-commit:
```bash
pixi run pre-commit install
```

The pre-commit configuration is already set up in `.pre-commit-config.yaml`. This includes:
* `ruff`: A fast Python linter and code formatter that will automatically fix issues when possible
* `black`: Python code formatting to ensure consistent code style
* `mypy`: Static type checking for Python to catch type-related errors before runtime

Pre-commit will now run automatically on every commit. If any checks fail, the commit will be aborted and the issues will be automatically fixed when possible.

## View and alter configurations

The project uses Hydra to manage configurations. You can view and modify these configurations from the command line.

To view available configurations:
```bash
pixi run python src/process.py --help
```

Output:

```yaml
process is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

model: model1, model2
process: process1, process2


== Config ==
Override anything in the config (foo.bar=value)

process:
  use_columns:
  - col1
  - col2
model:
  name: model1
data:
  raw: data/raw/sample.csv
  processed: data/processed/processed.csv
  final: data/final/final.csv
```

To override configurations (for example, changing the input data file):
```bash
pixi run python src/process.py data.raw=sample2.csv
```

Output:

```
Process data using sample2.csv
Columns used: ['col1', 'col2']
```

You can override any configuration value shown in the help output. Multiple overrides can be combined in a single command. For more information about Hydra's configuration system, visit the [official documentation](https://hydra.cc/docs/intro/).

## Auto-generate API documentation
Generate static documentation:
```bash
pixi run pdoc src -o docs
```

Start documentation server (available at http://localhost:8080):
```bash
pixi run pdoc src --http localhost:8080
```

The documentation will be generated from your docstrings and type hints in your Python files. The static documentation will be saved in the `docs` directory, while the live server allows you to view the documentation with hot-reloading as you make changes.
