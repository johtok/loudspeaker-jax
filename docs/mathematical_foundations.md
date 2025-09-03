# Mathematical Foundations of Loudspeaker System Identification

## Abstract

This document provides the mathematical foundations for the JAX-based loudspeaker system identification framework. We present the theoretical background for loudspeaker modeling, system identification methods, and the mathematical formulations underlying our implementation.

## 1. Loudspeaker Physical Model

### 1.1 Basic Loudspeaker Equations

The fundamental loudspeaker dynamics are described by the following system of differential equations:

```
L(x,i) di/dt + Re i + Bl(x) v = u(t)                    (1)
M dv/dt + Rm v + K(x) x = Bl(x) i + F_ext(t)           (2)
dx/dt = v                                               (3)
```

Where:
- `u(t)`: Input voltage [V]
- `i(t)`: Voice coil current [A]
- `x(t)`: Voice coil displacement [m]
- `v(t)`: Voice coil velocity [m/s]
- `L(x,i)`: Position and current-dependent inductance [H]
- `Re`: Electrical resistance [Ω]
- `Bl(x)`: Position-dependent force factor [N/A]
- `M`: Moving mass [kg]
- `Rm`: Mechanical resistance [N·s/m]
- `K(x)`: Position-dependent stiffness [N/m]
- `F_ext(t)`: External force [N]

### 1.2 Nonlinear Parameter Models

#### Force Factor Bl(x)
The force factor exhibits significant nonlinearity with displacement:

```
Bl(x) = Bl₀ + Bl₁x + Bl₂x² + Bl₃x³ + Bl₄x⁴           (4)
```

#### Suspension Stiffness K(x)
The suspension stiffness shows hardening/softening behavior:

```
K(x) = K₀ + K₁x + K₂x² + K₃x³ + K₄x⁴                 (5)
```

#### Inductance L(x,i)
The inductance depends on both position and current:

```
L(x,i) = L₀(x) · L₁(i)                                (6)

L₀(x) = L₀ + L₁x + L₂x² + L₃x³ + L₄x⁴                (7)
L₁(i) = 1 + Li₁i + Li₂i² + Li₃i³ + Li₄i⁴             (8)
```

### 1.3 Eddy Current Model (L2R2)

For high-frequency behavior, we include eddy current losses:

```
L₂(x,i) di₂/dt + R₂(x,i) i₂ = R₂(x,i) i               (9)
```

Where the eddy current parameters scale with the main inductance:

```
L₂(x,i) = L₂₀ · L(x,i)/L₀                            (10)
R₂(x,i) = R₂₀ · L(x,i)/L₀                            (11)
```

## 2. System Identification Methods

### 2.1 Linear System Identification

#### Cross-Spectral Density Matching
For linear parameter estimation, we minimize the cross-spectral density error:

```
J_linear = ∑ᵢ ||S_yx(ωᵢ) - Ĥ(ωᵢ)S_ux(ωᵢ)||²           (12)
```

Where:
- `S_yx(ω)`: Cross-spectral density between input and output
- `Ĥ(ω)`: Estimated transfer function
- `S_ux(ω)`: Input power spectral density

#### Subspace Identification
Using jax-sysid, we estimate state-space models:

```
x[k+1] = A x[k] + B u[k] + w[k]                       (13)
y[k] = C x[k] + D u[k] + v[k]                         (14)
```

### 2.2 Nonlinear System Identification

#### Gauss-Newton Optimization
For nonlinear parameter estimation, we use the Gauss-Newton method:

```
θ^(k+1) = θ^(k) - (J^T J)^(-1) J^T r                  (15)
```

Where:
- `θ`: Parameter vector
- `J`: Jacobian matrix of residuals
- `r`: Residual vector

#### Levenberg-Marquardt Algorithm
For robust convergence, we use the Levenberg-Marquardt modification:

```
θ^(k+1) = θ^(k) - (J^T J + μI)^(-1) J^T r             (16)
```

Where `μ` is the damping parameter.

### 2.3 Bayesian Parameter Inference

#### Probabilistic Model
We formulate the parameter estimation as a Bayesian inference problem:

```
p(θ|y) ∝ p(y|θ) p(θ)                                 (17)
```

Where:
- `p(θ|y)`: Posterior distribution
- `p(y|θ)`: Likelihood function
- `p(θ)`: Prior distribution

#### Hamiltonian Monte Carlo
For sampling from the posterior, we use HMC:

```
H(θ,p) = -log p(θ|y) + p^T M^(-1) p/2                 (18)
```

The dynamics follow:

```
dθ/dt = ∂H/∂p = M^(-1) p                             (19)
dp/dt = -∂H/∂θ = ∇_θ log p(θ|y)                      (20)
```

## 3. State-Space Modeling

### 3.1 Nonlinear State-Space Model

The loudspeaker can be represented as a nonlinear state-space model:

```
ẋ = f(x,u,θ) + w                                      (21)
y = h(x,u,θ) + v                                      (22)
```

Where:
- `x = [i, x, v, i₂]ᵀ`: State vector
- `u`: Input voltage
- `y = [i, v]ᵀ`: Output vector
- `w ~ N(0,Q)`: Process noise
- `v ~ N(0,R)`: Measurement noise

### 3.2 Extended Kalman Filter

For state estimation, we use the EKF:

```
Prediction:
x̂ₖ₋₁ = f(x̂ₖ₋₁, uₖ₋₁, θ)                              (23)
Pₖ₋₁ = Fₖ₋₁ Pₖ₋₁ Fₖ₋₁ᵀ + Q                           (24)

Update:
Kₖ = Pₖ₋₁ Hₖᵀ (Hₖ Pₖ₋₁ Hₖᵀ + R)^(-1)                 (25)
x̂ₖ = x̂ₖ₋₁ + Kₖ (yₖ - h(x̂ₖ₋₁, uₖ, θ))                (26)
Pₖ = (I - Kₖ Hₖ) Pₖ₋₁                                (27)
```

Where:
- `Fₖ = ∂f/∂x|ₓₖ`: State transition Jacobian
- `Hₖ = ∂h/∂x|ₓₖ`: Observation Jacobian

## 4. Gaussian Process Surrogates

### 4.1 GP Model for Unmodeled Nonlinearities

For capturing unmodeled effects, we use Gaussian Process regression:

```
f(x) ~ GP(m(x), k(x,x'))                              (28)
```

Where:
- `m(x)`: Mean function
- `k(x,x')`: Covariance function (kernel)

### 4.2 Hybrid Physics-Informed Model

We combine the physical model with GP surrogates:

```
ẋ = f_physics(x,u,θ) + f_GP(x,u)                      (29)
```

Where `f_GP` captures unmodeled nonlinearities.

### 4.3 Kernel Design

For loudspeaker applications, we use composite kernels:

```
k(x,x') = k_SE(x,x') + k_Periodic(x,x') + k_Linear(x,x')  (30)
```

Where:
- `k_SE`: Squared exponential kernel for smooth variations
- `k_Periodic`: Periodic kernel for harmonic distortions
- `k_Linear`: Linear kernel for trend components

## 5. Optimization Algorithms

### 5.1 Multi-Scale Optimization Strategy

We employ a hierarchical optimization approach:

1. **Linear Identification**: Estimate linear parameters using CSD matching
2. **Nonlinear Identification**: Estimate nonlinear parameters using Gauss-Newton
3. **Bayesian Refinement**: Sample posterior distributions using HMC

### 5.2 Regularization

To prevent overfitting, we use various regularization techniques:

#### L1 Regularization (Lasso)
```
J_reg = J_data + λ₁ ||θ||₁                            (31)
```

#### L2 Regularization (Ridge)
```
J_reg = J_data + λ₂ ||θ||₂²                           (32)
```

#### Group Lasso
```
J_reg = J_data + λ_g ∑_g ||θ_g||₂                     (33)
```

## 6. Uncertainty Quantification

### 6.1 Parameter Uncertainty

The posterior covariance matrix provides parameter uncertainties:

```
Σ_θ = (J^T J)^(-1) σ²                                 (34)
```

### 6.2 Prediction Uncertainty

For model predictions, we propagate uncertainties:

```
Var[y_pred] = H Σ_θ H^T + σ²                          (35)
```

### 6.3 Model Selection

We use information criteria for model selection:

#### Akaike Information Criterion (AIC)
```
AIC = -2 log L + 2k                                   (36)
```

#### Bayesian Information Criterion (BIC)
```
BIC = -2 log L + k log n                              (37)
```

Where:
- `L`: Likelihood
- `k`: Number of parameters
- `n`: Number of observations

## 7. Performance Metrics

### 7.1 Normalized Root Mean Square Error (NRMSE)

```
NRMSE = √(∑(y_true - y_pred)² / ∑(y_true - ȳ)²)      (38)
```

### 7.2 Coherence Function

```
C_xy(ω) = |S_xy(ω)|² / (S_xx(ω) S_yy(ω))             (39)
```

### 7.3 Total Harmonic Distortion (THD)

```
THD = √(∑(n=2 to ∞) A_n²) / A₁                       (40)
```

Where `A_n` are the harmonic amplitudes.

## 8. Implementation Considerations

### 8.1 Numerical Stability

- Use log-space parameterization for positive parameters
- Implement adaptive step sizes in ODE solvers
- Regularize ill-conditioned matrices

### 8.2 Computational Efficiency

- Exploit JAX's JIT compilation
- Use vectorized operations
- Implement efficient gradient computation

### 8.3 Memory Management

- Use checkpointing for long sequences
- Implement gradient accumulation
- Optimize data loading and preprocessing

## References

1. Klippel, W. (2003). Dynamic measurement and interpretation of the nonlinear parameters of electrodynamic loudspeakers. *Journal of the Audio Engineering Society*, 51(12), 944-955.

2. Vanderkooy, J. (1989). A model of loudspeaker driver impedance incorporating eddy currents in the pole structure. *Journal of the Audio Engineering Society*, 37(3), 119-128.

3. Levenberg, K. (1944). A method for the solution of certain non-linear problems in least squares. *Quarterly of Applied Mathematics*, 2(2), 164-168.

4. Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters. *Journal of the Society for Industrial and Applied Mathematics*, 11(2), 431-441.

5. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian processes for machine learning*. MIT Press.

6. Särkkä, S. (2013). *Bayesian filtering and smoothing*. Cambridge University Press.
