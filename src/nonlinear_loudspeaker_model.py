"""
Nonlinear Loudspeaker Model Implementation.

This module implements a comprehensive nonlinear loudspeaker model using JAX and Diffrax,
following the methodology from Heuchel et al. ICA 2022.

Author: Research Team
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import diffrax as dfx
from equinox import Module, field
from jaxopt import GaussNewton, LevenbergMarquardt
import optax

from ground_truth_model import GroundTruthLoudspeakerModel


class NonlinearLoudspeakerModel(Module):
    """
    Nonlinear loudspeaker model with full JAX/Diffrax integration.
    
    This model implements the complete nonlinear loudspeaker dynamics including:
    - Bl(x): Force factor nonlinearity
    - K(x): Stiffness nonlinearity  
    - L(x,i): Inductance nonlinearity (position and current dependent)
    - Eddy current model (L2R2)
    
    State vector: [i, x, v, i2] where:
    - i: voice coil current [A]
    - x: voice coil displacement [m]
    - v: voice coil velocity [m/s]
    - i2: eddy current [A]
    """
    
    # Linear parameters
    Re: float = field(default=6.8, static=True)      # Electrical resistance [Ω]
    Le: float = field(default=0.5e-3, static=True)   # Electrical inductance [H]
    Bl: float = field(default=3.2, static=True)      # Force factor [N/A]
    M: float = field(default=12e-3, static=True)     # Moving mass [kg]
    K: float = field(default=1200, static=True)      # Stiffness [N/m]
    Rm: float = field(default=0.8, static=True)      # Mechanical resistance [N·s/m]
    
    # Eddy current parameters
    L20: float = field(default=0.1e-3, static=True)  # Eddy current inductance [H]
    R20: float = field(default=0.5, static=True)     # Eddy current resistance [Ω]
    
    # Nonlinear parameters (polynomial coefficients)
    Bl_nl: jnp.ndarray = field(default_factory=lambda: jnp.zeros(4))
    K_nl: jnp.ndarray = field(default_factory=lambda: jnp.zeros(4))
    L_nl: jnp.ndarray = field(default_factory=lambda: jnp.zeros(4))
    Li_nl: jnp.ndarray = field(default_factory=lambda: jnp.zeros(4))
    
    def force_factor(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate position-dependent force factor Bl(x)."""
        return self.Bl + jnp.polyval(jnp.append(self.Bl_nl, 0), x)
    
    def stiffness(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate position-dependent stiffness K(x)."""
        return self.K + jnp.polyval(jnp.append(self.K_nl, 0), x)
    
    def inductance(self, x: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
        """Calculate position and current-dependent inductance L(x,i)."""
        L0 = self.Le + jnp.polyval(jnp.append(self.L_nl, 0), x)
        Li = 1.0 + jnp.polyval(jnp.append(self.Li_nl, 0), i)
        return L0 * Li
    
    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate state derivatives for the nonlinear loudspeaker model.
        
        Args:
            x: State vector [i, x, v, i2]
            u: Input voltage [V]
            
        Returns:
            State derivatives [di/dt, dx/dt, dv/dt, di2/dt]
        """
        i, x_pos, v, i2 = x
        
        # Calculate nonlinear parameters
        Bl = self.force_factor(x_pos)
        K = self.stiffness(x_pos)
        L = self.inductance(x_pos, i)
        
        # Calculate partial derivatives of inductance
        L_dx = jax.grad(self.inductance, argnums=0)(x_pos, i)
        L_di = jax.grad(self.inductance, argnums=1)(x_pos, i)
        
        # Eddy current parameters (scale with main inductance)
        L2 = self.L20 * L / self.Le
        R2 = self.R20 * L / self.Le
        L2_dx = self.L20 * L_dx / self.Le
        L2_di = self.L20 * L_di / self.Le
        
        # State derivatives following Heuchel et al. formulation
        # Electrical equation: L di/dt + Re i + Bl v = u - R2(i - i2)
        di_dt = (u - self.Re * i - R2 * (i - i2) - (Bl + L_dx * i) * v) / (L + i * L_di)
        
        # Kinematic equation: dx/dt = v
        dx_dt = v
        
        # Mechanical equation: M dv/dt + Rm v + K x = Bl i + 0.5*L_dx*i^2
        dv_dt = ((Bl + 0.5 * (L_dx * i + L2_dx * i2)) * i - self.Rm * v - K * x_pos) / self.M
        
        # Eddy current equation: L2 di2/dt + R2 i2 = R2 i - L2_dx*i2*v
        di2_dt = (R2 * (i - i2) - L2_dx * i2 * v) / (L2 + i2 * L2_di)
        
        return jnp.array([di_dt, dx_dt, dv_dt, di2_dt])
    
    def output(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Calculate model outputs [current, velocity]."""
        i, _, v, _ = x
        return jnp.array([i, v])
    
    def simulate(self, u: jnp.ndarray, x0: jnp.ndarray, 
                 dt: float = 1e-4, solver: str = 'Tsit5') -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate the loudspeaker model using Diffrax.
        
        Args:
            u: Input voltage time series [V]
            x0: Initial state [i, x, v, i2]
            dt: Time step [s]
            solver: ODE solver ('Tsit5', 'Dopri5', 'Heun')
            
        Returns:
            Tuple of (time_vector, state_trajectory)
        """
        # Create time vector
        n_samples = len(u)
        t = jnp.linspace(0, (n_samples - 1) * dt, n_samples)
        
        # Define ODE function with proper input handling
        def ode_func(t, y, args):
            # Find the closest time index for input interpolation
            idx = jnp.clip(jnp.round(t / dt).astype(int), 0, len(u) - 1)
            u_interp = u[idx]
            return self.dynamics(y, u_interp)
        
        # Choose solver
        if solver == 'Tsit5':
            ode_solver = dfx.Tsit5()
        elif solver == 'Dopri5':
            ode_solver = dfx.Dopri5()
        elif solver == 'Heun':
            ode_solver = dfx.Heun()
        else:
            raise ValueError(f"Unknown solver: {solver}")
        
        # Solve ODE
        solution = dfx.diffeqsolve(
            dfx.ODETerm(ode_func),
            ode_solver,
            t[0], t[-1], dt0=dt,
            y0=x0,
            args=None,
            saveat=dfx.SaveAt(ts=t)
        )
        
        return t, solution.ys
    
    def predict(self, u: jnp.ndarray, x0: jnp.ndarray = None, 
                dt: float = 1e-4) -> jnp.ndarray:
        """
        Predict model outputs for given input.
        
        Args:
            u: Input voltage time series [V]
            x0: Initial state (default: zeros)
            dt: Time step [s]
            
        Returns:
            Predicted outputs [current, velocity]
        """
        if x0 is None:
            x0 = jnp.zeros(4)
        
        t, x_traj = self.simulate(u, x0, dt)
        outputs = jnp.array([self.output(x, u[i]) for i, x in enumerate(x_traj)])
        return outputs
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all model parameters as a dictionary."""
        return {
            'Re': self.Re,
            'Le': self.Le,
            'Bl': self.Bl,
            'M': self.M,
            'K': self.K,
            'Rm': self.Rm,
            'L20': self.L20,
            'R20': self.R20,
            'Bl_nl': self.Bl_nl,
            'K_nl': self.K_nl,
            'L_nl': self.L_nl,
            'Li_nl': self.Li_nl
        }
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set model parameters from dictionary."""
        # Get current parameters and update them
        current_params = self.get_parameters()
        current_params.update(params)
        
        # Create new model instance with updated parameters
        # This is the correct way to handle Equinox modules
        return NonlinearLoudspeakerModel(
            Re=current_params['Re'],
            Le=current_params['Le'],
            Bl=current_params['Bl'],
            M=current_params['M'],
            K=current_params['K'],
            Rm=current_params['Rm'],
            L20=current_params['L20'],
            R20=current_params['R20'],
            Bl_nl=current_params['Bl_nl'],
            K_nl=current_params['K_nl'],
            L_nl=current_params['L_nl'],
            Li_nl=current_params['Li_nl']
        )


class NonlinearSystemIdentifier:
    """
    System identification for nonlinear loudspeaker models.
    
    Implements various optimization algorithms for parameter estimation:
    - Gauss-Newton
    - Levenberg-Marquardt
    - L-BFGS
    """
    
    def __init__(self, model: NonlinearLoudspeakerModel):
        """Initialize identifier with model."""
        self.model = model
        self.optimizer = None
        self.optimization_state = None
    
    def loss_function(self, params: Dict[str, Any], u: jnp.ndarray, 
                     y_measured: jnp.ndarray, x0: jnp.ndarray = None) -> float:
        """
        Calculate loss function for parameter estimation.
        
        Args:
            params: Model parameters
            u: Input voltage [V]
            y_measured: Measured outputs [current, velocity]
            x0: Initial state
            
        Returns:
            Mean squared error loss
        """
        # Create model with new parameters
        model = self.model.set_parameters(params)
        
        # Predict outputs
        y_pred = model.predict(u, x0)
        
        # Calculate MSE loss
        loss = jnp.mean((y_measured - y_pred) ** 2)
        return loss
    
    def gauss_newton_optimization(self, u: jnp.ndarray, y_measured: jnp.ndarray,
                                 initial_params: Dict[str, Any], 
                                 x0: jnp.ndarray = None,
                                 max_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimize parameters using Gauss-Newton method.
        
        Args:
            u: Input voltage [V]
            y_measured: Measured outputs [current, velocity]
            initial_params: Initial parameter values
            x0: Initial state
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        # Create loss function
        def loss_fn(params):
            return self.loss_function(params, u, y_measured, x0)
        
        # Initialize Gauss-Newton optimizer
        gn = GaussNewton(loss_fn, maxiter=max_iterations)
        
        # Run optimization
        result = gn.run(initial_params)
        
        return {
            'parameters': result.params,
            'loss': result.state.value,
            'iterations': result.state.iter_num,
            'converged': result.state.converged
        }
    
    def levenberg_marquardt_optimization(self, u: jnp.ndarray, y_measured: jnp.ndarray,
                                        initial_params: Dict[str, Any],
                                        x0: jnp.ndarray = None,
                                        max_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimize parameters using Levenberg-Marquardt method.
        
        Args:
            u: Input voltage [V]
            y_measured: Measured outputs [current, velocity]
            initial_params: Initial parameter values
            x0: Initial state
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        # Create loss function
        def loss_fn(params):
            return self.loss_function(params, u, y_measured, x0)
        
        # Initialize Levenberg-Marquardt optimizer
        lm = LevenbergMarquardt(loss_fn, maxiter=max_iterations)
        
        # Run optimization
        result = lm.run(initial_params)
        
        return {
            'parameters': result.params,
            'loss': result.state.value,
            'iterations': result.state.iter_num,
            'converged': result.state.converged
        }
    
    def lbfgs_optimization(self, u: jnp.ndarray, y_measured: jnp.ndarray,
                          initial_params: Dict[str, Any],
                          x0: jnp.ndarray = None,
                          max_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimize parameters using L-BFGS method.
        
        Args:
            u: Input voltage [V]
            y_measured: Measured outputs [current, velocity]
            initial_params: Initial parameter values
            x0: Initial state
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        from jaxopt import LBFGS
        
        # Create loss function
        def loss_fn(params):
            return self.loss_function(params, u, y_measured, x0)
        
        # Initialize L-BFGS optimizer
        lbfgs = LBFGS(loss_fn, maxiter=max_iterations)
        
        # Run optimization
        result = lbfgs.run(initial_params)
        
        return {
            'parameters': result.params,
            'loss': result.state.value,
            'iterations': result.state.iter_num,
            'converged': result.state.converged
        }


def create_initial_parameters() -> Dict[str, Any]:
    """Create initial parameter values for optimization."""
    return {
        'Re': 6.8,
        'Le': 0.5e-3,
        'Bl': 3.2,
        'M': 12e-3,
        'K': 1200,
        'Rm': 0.8,
        'L20': 0.1e-3,
        'R20': 0.5,
        'Bl_nl': jnp.array([0.0, 0.0, -50.0, -0.1]),
        'K_nl': jnp.array([0.0, 0.0, 100.0, 0.0]),
        'L_nl': jnp.array([0.0, 0.0, 0.0, 0.0]),
        'Li_nl': jnp.array([0.0, 0.0, 0.0, 0.0])
    }


def nonlinear_identification_method(u: jnp.ndarray, y: jnp.ndarray, 
                                  method: str = 'gauss_newton',
                                  **kwargs) -> Dict[str, Any]:
    """
    Nonlinear system identification method.
    
    Args:
        u: Input voltage [V]
        y: Output measurements [current, velocity]
        method: Optimization method ('gauss_newton', 'levenberg_marquardt', 'lbfgs')
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with identification results
    """
    print(f"Running nonlinear identification using {method}...")
    
    # Create model
    model = NonlinearLoudspeakerModel()
    
    # Create identifier
    identifier = NonlinearSystemIdentifier(model)
    
    # Get initial parameters
    initial_params = kwargs.get('initial_params', create_initial_params())
    max_iterations = kwargs.get('max_iterations', 100)
    
    # Run optimization
    if method == 'gauss_newton':
        result = identifier.gauss_newton_optimization(
            u, y, initial_params, max_iterations=max_iterations
        )
    elif method == 'levenberg_marquardt':
        result = identifier.levenberg_marquardt_optimization(
            u, y, initial_params, max_iterations=max_iterations
        )
    elif method == 'lbfgs':
        result = identifier.lbfgs_optimization(
            u, y, initial_params, max_iterations=max_iterations
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    # Generate predictions with fitted parameters
    model.set_parameters(result['parameters'])
    predictions = model.predict(u)
    
    return {
        'model': model,
        'parameters': result['parameters'],
        'predictions': predictions,
        'convergence': {
            'converged': result['converged'],
            'iterations': result['iterations'],
            'final_loss': result['loss'],
            'method': method
        }
    }
