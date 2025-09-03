"""
Loudspeaker Physical Model Implementation.

This module implements a comprehensive loudspeaker physical model with nonlinearities,
following the mathematical foundations outlined in the documentation.

Author: Research Team
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional
from dataclasses import dataclass, replace
import warnings


@dataclass
class LoudspeakerModel:
    """
    Comprehensive loudspeaker physical model with nonlinearities.
    
    This model implements the full loudspeaker dynamics including:
    - Nonlinear force factor Bl(x)
    - Nonlinear suspension stiffness K(x)
    - Current and position dependent inductance L(x,i)
    - Eddy current losses (L2R2 model)
    
    State vector: [i, x, v, i2] where:
    - i: voice coil current [A]
    - x: voice coil displacement [m]
    - v: voice coil velocity [m/s]
    - i2: eddy current [A]
    """
    
    # Linear parameters
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
    Bl_nl: jnp.ndarray = None  # Bl(x) polynomial coefficients
    K_nl: jnp.ndarray = None   # K(x) polynomial coefficients
    L_nl: jnp.ndarray = None   # L(x) polynomial coefficients
    Li_nl: jnp.ndarray = None  # L(i) polynomial coefficients
    
    # Model dimensions
    n_states: int = 4    # [i, x, v, i2]
    n_inputs: int = 1    # voltage
    n_outputs: int = 2   # [current, velocity]
    
    def __post_init__(self):
        """Initialize default nonlinear parameters if not provided."""
        if self.Bl_nl is None:
            self.Bl_nl = jnp.zeros(4)
        if self.K_nl is None:
            self.K_nl = jnp.zeros(4)
        if self.L_nl is None:
            self.L_nl = jnp.zeros(4)
        if self.Li_nl is None:
            self.Li_nl = jnp.zeros(4)
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate that all parameters are physically reasonable."""
        # Check positive parameters
        positive_params = {
            'Re': self.Re, 'Le': self.Le, 'Bl': self.Bl,
            'M': self.M, 'K': self.K, 'Rm': self.Rm,
            'L20': self.L20, 'R20': self.R20
        }
        
        for name, value in positive_params.items():
            if value <= 0:
                raise ValueError(f"Parameter {name} must be positive, got {value}")
        
        # Check reasonable ranges
        if not (1e-3 < self.M < 100e-3):
            warnings.warn(f"Moving mass M={self.M} may be outside typical range [1e-3, 100e-3] kg")
        
        if not (100 < self.K < 10000):
            warnings.warn(f"Stiffness K={self.K} may be outside typical range [100, 10000] N/m")
        
        if not (0.5 < self.Bl < 20):
            warnings.warn(f"Force factor Bl={self.Bl} may be outside typical range [0.5, 20] N/A")
    
    def force_factor(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate position-dependent force factor Bl(x).
        
        Bl(x) = Bl + Bl_nl[0]*x^4 + Bl_nl[1]*x^3 + Bl_nl[2]*x^2 + Bl_nl[3]*x
        
        Args:
            x: Displacement [m]
            
        Returns:
            Force factor [N/A]
        """
        # Polynomial: Bl + Bl_nl[0]*x^4 + Bl_nl[1]*x^3 + Bl_nl[2]*x^2 + Bl_nl[3]*x
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
        # Polynomial: K + K_nl[0]*x^4 + K_nl[1]*x^3 + K_nl[2]*x^2 + K_nl[3]*x
        return self.K + jnp.polyval(jnp.append(self.K_nl, 0), x)
    
    def inductance(self, x: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate position and current-dependent inductance L(x,i).
        
        L(x,i) = L₀(x) * L₁(i)
        L₀(x) = Le + L_nl[0]*x^4 + L_nl[1]*x^3 + L_nl[2]*x^2 + L_nl[3]*x
        L₁(i) = 1 + Li_nl[0]*i^4 + Li_nl[1]*i^3 + Li_nl[2]*i^2 + Li_nl[3]*i
        
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
        Calculate state derivatives for the loudspeaker model.
        
        State vector: [i, x, v, i2]
        Input: u (voltage)
        
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
        
        # State derivatives
        # Electrical equation: L di/dt + Re i + Bl v = u
        di_dt = (u - self.Re * i - R2 * (i - i2) - (Bl + L_dx * i) * v) / (L + i * L_di)
        
        # Kinematic equation: dx/dt = v
        dx_dt = v
        
        # Mechanical equation: M dv/dt + Rm v + K x = Bl i
        dv_dt = ((Bl + 0.5 * (L_dx * i + L2_dx * i2)) * i - self.Rm * v - K * x_pos) / self.M
        
        # Eddy current equation: L2 di2/dt + R2 i2 = R2 i
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
    
    def replace(self, **kwargs) -> 'LoudspeakerModel':
        """Create a new model with some parameters replaced."""
        return replace(self, **kwargs)
    
    def to_linear(self) -> 'LoudspeakerModel':
        """Create a linear version of the model (no nonlinearities)."""
        return replace(
            self,
            Bl_nl=jnp.zeros(4),
            K_nl=jnp.zeros(4),
            L_nl=jnp.zeros(4),
            Li_nl=jnp.zeros(4)
        )
    
    def get_linear_parameters(self) -> Dict[str, float]:
        """Get only the linear parameters."""
        return {
            'Re': self.Re,
            'Le': self.Le,
            'Bl': self.Bl,
            'M': self.M,
            'K': self.K,
            'Rm': self.Rm
        }
    
    def get_nonlinear_parameters(self) -> Dict[str, jnp.ndarray]:
        """Get only the nonlinear parameters."""
        return {
            'Bl_nl': self.Bl_nl,
            'K_nl': self.K_nl,
            'L_nl': self.L_nl,
            'Li_nl': self.Li_nl
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"LoudspeakerModel(Re={self.Re}, Le={self.Le}, Bl={self.Bl}, "
                f"M={self.M}, K={self.K}, Rm={self.Rm})")


# Utility functions for model analysis

def create_standard_model() -> LoudspeakerModel:
    """Create a standard loudspeaker model with typical parameters."""
    return LoudspeakerModel(
        Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8,
        L20=0.1e-3, R20=0.5
    )


def create_nonlinear_model() -> LoudspeakerModel:
    """Create a nonlinear loudspeaker model with typical nonlinearities."""
    return LoudspeakerModel(
        Re=6.8, Le=0.5e-3, Bl=3.2, M=12e-3, K=1200, Rm=0.8,
        L20=0.1e-3, R20=0.5,
        Bl_nl=jnp.array([0.0, -0.1, 0.0, 0.0]),
        K_nl=jnp.array([0.0, 0.0, 100.0, 0.0]),
        L_nl=jnp.array([0.0, 0.0, 0.0, 0.0]),
        Li_nl=jnp.array([0.0, 0.0, 0.0, 0.0])
    )


def analyze_model_linearity(model: LoudspeakerModel, 
                          x_range: jnp.ndarray = jnp.linspace(-0.005, 0.005, 100),
                          i_range: jnp.ndarray = jnp.linspace(-2, 2, 100)) -> Dict[str, jnp.ndarray]:
    """
    Analyze the linearity of the model over specified ranges.
    
    Args:
        model: Loudspeaker model
        x_range: Displacement range for analysis [m]
        i_range: Current range for analysis [A]
        
    Returns:
        Dictionary with linearity analysis results
    """
    # Analyze Bl(x) linearity
    Bl_values = jnp.array([model.force_factor(x) for x in x_range])
    Bl_deviation = Bl_values - model.Bl
    
    # Analyze K(x) linearity
    K_values = jnp.array([model.stiffness(x) for x in x_range])
    K_deviation = K_values - model.K
    
    # Analyze L(x,i) linearity
    L_values = jnp.array([[model.inductance(x, i) for x in x_range] for i in i_range])
    L_deviation = L_values - model.Le
    
    return {
        'x_range': x_range,
        'i_range': i_range,
        'Bl_values': Bl_values,
        'Bl_deviation': Bl_deviation,
        'K_values': K_values,
        'K_deviation': K_deviation,
        'L_values': L_values,
        'L_deviation': L_deviation,
        'max_Bl_deviation': jnp.max(jnp.abs(Bl_deviation)),
        'max_K_deviation': jnp.max(jnp.abs(K_deviation)),
        'max_L_deviation': jnp.max(jnp.abs(L_deviation))
    }
