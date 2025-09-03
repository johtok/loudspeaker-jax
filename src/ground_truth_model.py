"""
Ground Truth Nonlinear Loudspeaker Model.

This module implements the ground truth nonlinear loudspeaker model based on
the reference implementation from Heuchel et al. ICA 2022 paper:
"A quantitative comparison of linear and nonlinear loudspeaker models"

Reference: https://github.com/fhchl/quant-comp-ls-mod-ica22

Author: Research Team
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Any
from dataclasses import dataclass
import diffrax as dfx


@dataclass
class GroundTruthLoudspeakerModel:
    """
    Ground truth nonlinear loudspeaker model based on Heuchel et al. ICA 2022.
    
    This model serves as the reference for generating synthetic data and
    validating system identification methods.
    
    State vector: [i, x, v, i2] where:
    - i: voice coil current [A]
    - x: voice coil displacement [m]
    - v: voice coil velocity [m/s]
    - i2: eddy current [A]
    """
    
    # Linear parameters (typical values from literature)
    Re: float = 6.8      # Electrical resistance [Ω]
    Le: float = 0.5e-3   # Electrical inductance [H]
    Bl: float = 3.2      # Force factor [N/A]
    M: float = 12e-3     # Moving mass [kg]
    K: float = 1200      # Stiffness [N/m]
    Rm: float = 0.8      # Mechanical resistance [N·s/m]
    
    # Eddy current parameters (L2R2 model)
    L20: float = 0.1e-3  # Eddy current inductance [H]
    R20: float = 0.5     # Eddy current resistance [Ω]
    
    # Nonlinear parameters (polynomial coefficients)
    # Bl(x) = Bl + Bl_nl[0]*x^4 + Bl_nl[1]*x^3 + Bl_nl[2]*x^2 + Bl_nl[3]*x
    Bl_nl: jnp.ndarray = None  # Force factor nonlinearity
    
    # K(x) = K + K_nl[0]*x^4 + K_nl[1]*x^3 + K_nl[2]*x^2 + K_nl[3]*x
    K_nl: jnp.ndarray = None   # Stiffness nonlinearity
    
    # L(x,i) = L0(x) * L1(i)
    # L0(x) = Le + L_nl[0]*x^4 + L_nl[1]*x^3 + L_nl[2]*x^2 + L_nl[3]*x
    L_nl: jnp.ndarray = None   # Inductance position nonlinearity
    
    # L1(i) = 1 + Li_nl[0]*i^4 + Li_nl[1]*i^3 + Li_nl[2]*i^2 + Li_nl[3]*i
    Li_nl: jnp.ndarray = None  # Inductance current nonlinearity
    
    # Model dimensions
    n_states: int = 4    # [i, x, v, i2]
    n_inputs: int = 1    # voltage
    n_outputs: int = 2   # [current, velocity]
    
    def __post_init__(self):
        """Initialize default nonlinear parameters if not provided."""
        if self.Bl_nl is None:
            # Typical nonlinear force factor (softening at large displacements)
            self.Bl_nl = jnp.array([0.0, 0.0, -50.0, -0.1])
        
        if self.K_nl is None:
            # Typical nonlinear stiffness (hardening at large displacements)
            self.K_nl = jnp.array([0.0, 0.0, 100.0, 0.0])
        
        if self.L_nl is None:
            # Typical nonlinear inductance (position dependent)
            self.L_nl = jnp.array([0.0, 0.0, 0.0, 0.0])
        
        if self.Li_nl is None:
            # Typical nonlinear inductance (current dependent)
            self.Li_nl = jnp.array([0.0, 0.0, 0.0, 0.0])
    
    def force_factor(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate position-dependent force factor Bl(x).
        
        Bl(x) = Bl + Bl_nl[0]*x^4 + Bl_nl[1]*x^3 + Bl_nl[2]*x^2 + Bl_nl[3]*x
        
        Args:
            x: Displacement [m]
            
        Returns:
            Force factor [N/A]
        """
        return self.Bl + jnp.polyval(jnp.append(self.Bl_nl, 0), x)
    
    def stiffness(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate position-dependent stiffness K(x).
        
        K(x) = K + K_nl[0]*x^4 + K_nl[1]*x^3 + K_nl[2]*x^2 + K_nl[3]*x
        
        Args:
            x: Displacement [m]
            
        Returns:
            Stiffness [N/m]
        """
        return self.K + jnp.polyval(jnp.append(self.K_nl, 0), x)
    
    def inductance(self, x: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate position and current-dependent inductance L(x,i).
        
        L(x,i) = L0(x) * L1(i)
        L0(x) = Le + L_nl[0]*x^4 + L_nl[1]*x^3 + L_nl[2]*x^2 + L_nl[3]*x
        L1(i) = 1 + Li_nl[0]*i^4 + Li_nl[1]*i^3 + Li_nl[2]*i^2 + Li_nl[3]*i
        
        Args:
            x: Displacement [m]
            i: Current [A]
            
        Returns:
            Inductance [H]
        """
        # Position-dependent part
        L0 = self.Le + jnp.polyval(jnp.append(self.L_nl, 0), x)
        
        # Current-dependent part
        Li = 1.0 + jnp.polyval(jnp.append(self.Li_nl, 0), i)
        
        return L0 * Li
    
    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate state derivatives for the nonlinear loudspeaker model.
        
        Based on Heuchel et al. ICA 2022 implementation.
        
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
        """
        Calculate model outputs.
        
        Args:
            x: State vector [i, x, v, i2]
            u: Input voltage [V]
            
        Returns:
            Output vector [current, velocity]
        """
        i, _, v, _ = x
        return jnp.array([i, v])
    
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
    
    def generate_synthetic_data(self, u: jnp.ndarray, x0: jnp.ndarray = None,
                               dt: float = 1e-4, noise_level: float = 0.01) -> Dict[str, jnp.ndarray]:
        """
        Generate synthetic measurement data with noise.
        
        Args:
            u: Input voltage time series [V]
            x0: Initial state (default: zeros)
            dt: Time step [s]
            noise_level: Relative noise level (std/mean)
            
        Returns:
            Dictionary with synthetic data
        """
        if x0 is None:
            x0 = jnp.zeros(4)
        
        # Simulate clean response
        t, x_traj = self.simulate(u, x0, dt)
        
        # Extract outputs
        i_clean = x_traj[:, 0]  # Current
        v_clean = x_traj[:, 2]  # Velocity
        
        # Add measurement noise
        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)
        i_noise = jax.random.normal(subkey, i_clean.shape) * noise_level * jnp.std(i_clean)
        
        rng_key, subkey = jax.random.split(rng_key)
        v_noise = jax.random.normal(subkey, v_clean.shape) * noise_level * jnp.std(v_clean)
        
        i_measured = i_clean + i_noise
        v_measured = v_clean + v_noise
        
        return {
            'time': t,
            'voltage': u,
            'current_clean': i_clean,
            'velocity_clean': v_clean,
            'current_measured': i_measured,
            'velocity_measured': v_measured,
            'current_noise': i_noise,
            'velocity_noise': v_noise,
            'state_trajectory': x_traj
        }


def create_standard_ground_truth() -> GroundTruthLoudspeakerModel:
    """Create a standard ground truth model with typical parameters."""
    return GroundTruthLoudspeakerModel(
        Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8,
        L20=0.1e-3, R20=0.5,
        Bl_nl=jnp.array([0.0, 0.0, -50.0, -0.1]),
        K_nl=jnp.array([0.0, 0.0, 100.0, 0.0]),
        L_nl=jnp.array([0.0, 0.0, 0.0, 0.0]),
        Li_nl=jnp.array([0.0, 0.0, 0.0, 0.0])
    )


def create_highly_nonlinear_ground_truth() -> GroundTruthLoudspeakerModel:
    """Create a highly nonlinear ground truth model for testing."""
    return GroundTruthLoudspeakerModel(
        Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8,
        L20=0.1e-3, R20=0.5,
        Bl_nl=jnp.array([0.0, 0.0, -100.0, -0.2]),
        K_nl=jnp.array([0.0, 0.0, 200.0, 0.0]),
        L_nl=jnp.array([0.0, 0.0, 0.0, 0.0]),
        Li_nl=jnp.array([0.0, 0.0, 0.0, 0.0])
    )


def create_linear_ground_truth() -> GroundTruthLoudspeakerModel:
    """Create a linear ground truth model (no nonlinearities)."""
    return GroundTruthLoudspeakerModel(
        Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8,
        L20=0.1e-3, R20=0.5,
        Bl_nl=jnp.zeros(4),
        K_nl=jnp.zeros(4),
        L_nl=jnp.zeros(4),
        Li_nl=jnp.zeros(4)
    )
