"""
State-Space Modeling for Loudspeaker System Identification using Dynamax.

This module implements state-space models using Dynamax for:
- Linear Dynamical Systems (LDS)
- Nonlinear State-Space Models
- Probabilistic state estimation

Author: Research Team
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import dynamax
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.nonlinear_gaussian_ssm import NonlinearGaussianSSM
from dynamax.parameters import LGSSMParams, NLGSSMParams
from dynamax.inference import lgssm_smoother, nlssm_smoother
from dynamax.utils import ensure_array_has_batch_dim

from nonlinear_loudspeaker_model import NonlinearLoudspeakerModel


class LoudspeakerStateSpaceModel:
    """
    State-space model for loudspeaker system identification using Dynamax.
    
    Implements both linear and nonlinear state-space representations.
    """
    
    def __init__(self, model_type: str = 'linear'):
        """
        Initialize state-space model.
        
        Args:
            model_type: Type of model ('linear', 'nonlinear')
        """
        self.model_type = model_type
        self.ssm = None
        self.params = None
        self.trained = False
    
    def create_linear_ssm(self, state_dim: int = 4, emission_dim: int = 2,
                         input_dim: int = 1) -> LinearGaussianSSM:
        """
        Create linear Gaussian state-space model.
        
        Args:
            state_dim: State dimension (4 for [i, x, v, i2])
            emission_dim: Emission dimension (2 for [current, velocity])
            input_dim: Input dimension (1 for voltage)
            
        Returns:
            LinearGaussianSSM object
        """
        # Create linear SSM
        ssm = LinearGaussianSSM(state_dim, emission_dim, input_dim)
        
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        params = ssm.initialize_params(key)
        
        # Set reasonable initial values for loudspeaker
        params.dynamics.weights = jnp.eye(state_dim) * 0.9  # Slightly damped
        params.dynamics.input_weights = jnp.zeros((state_dim, input_dim))
        params.dynamics.input_weights = params.dynamics.input_weights.at[0, 0].set(1.0)  # Input affects current
        
        params.emissions.weights = jnp.zeros((emission_dim, state_dim))
        params.emissions.weights = params.emissions.weights.at[0, 0].set(1.0)  # Current observation
        params.emissions.weights = params.emissions.weights.at[1, 2].set(1.0)  # Velocity observation
        
        # Set noise covariances
        params.dynamics.cov = jnp.eye(state_dim) * 0.01
        params.emissions.cov = jnp.eye(emission_dim) * 0.1
        
        self.ssm = ssm
        self.params = params
        
        return ssm
    
    def create_nonlinear_ssm(self, state_dim: int = 4, emission_dim: int = 2,
                            input_dim: int = 1) -> NonlinearGaussianSSM:
        """
        Create nonlinear Gaussian state-space model.
        
        Args:
            state_dim: State dimension (4 for [i, x, v, i2])
            emission_dim: Emission dimension (2 for [current, velocity])
            input_dim: Input dimension (1 for voltage)
            
        Returns:
            NonlinearGaussianSSM object
        """
        # Define nonlinear dynamics function
        def dynamics_fn(state, input, params):
            """Nonlinear dynamics for loudspeaker model."""
            i, x, v, i2 = state
            u = input[0] if input.ndim > 0 else input
            
            # Simplified nonlinear dynamics
            di_dt = (u - 6.8 * i - 3.2 * v) / 0.5e-3
            dx_dt = v
            dv_dt = (3.2 * i - 0.8 * v - 1200 * x) / 12e-3
            di2_dt = (0.5 * (i - i2)) / 0.1e-3
            
            return jnp.array([di_dt, dx_dt, dv_dt, di2_dt])
        
        # Define emission function
        def emission_fn(state, input, params):
            """Emission function for loudspeaker model."""
            i, x, v, i2 = state
            return jnp.array([i, v])  # Observe current and velocity
        
        # Create nonlinear SSM
        ssm = NonlinearGaussianSSM(
            dynamics_fn=dynamics_fn,
            emission_fn=emission_fn,
            state_dim=state_dim,
            emission_dim=emission_dim,
            input_dim=input_dim
        )
        
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        params = ssm.initialize_params(key)
        
        # Set noise covariances
        params.dynamics.cov = jnp.eye(state_dim) * 0.01
        params.emissions.cov = jnp.eye(emission_dim) * 0.1
        
        self.ssm = ssm
        self.params = params
        
        return ssm
    
    def fit_em_algorithm(self, inputs: jnp.ndarray, emissions: jnp.ndarray,
                        num_iters: int = 100) -> Dict[str, Any]:
        """
        Fit state-space model using EM algorithm.
        
        Args:
            inputs: Input sequence [T, input_dim]
            emissions: Emission sequence [T, emission_dim]
            num_iters: Number of EM iterations
            
        Returns:
            Dictionary with fitting results
        """
        print(f"Fitting {self.model_type} state-space model using EM...")
        
        if self.ssm is None:
            if self.model_type == 'linear':
                self.create_linear_ssm()
            else:
                self.create_nonlinear_ssm()
        
        # Ensure inputs and emissions have batch dimensions
        inputs = ensure_array_has_batch_dim(inputs)
        emissions = ensure_array_has_batch_dim(emissions)
        
        # Run EM algorithm
        fitted_params, log_likelihoods = self.ssm.fit_em(
            self.params, inputs, emissions, num_iters=num_iters
        )
        
        self.params = fitted_params
        self.trained = True
        
        return {
            'fitted_params': fitted_params,
            'log_likelihoods': log_likelihoods,
            'final_log_likelihood': log_likelihoods[-1]
        }
    
    def smooth(self, inputs: jnp.ndarray, emissions: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Perform smoothing to estimate states.
        
        Args:
            inputs: Input sequence [T, input_dim]
            emissions: Emission sequence [T, emission_dim]
            
        Returns:
            Dictionary with smoothing results
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit_em_algorithm first.")
        
        # Ensure inputs and emissions have batch dimensions
        inputs = ensure_array_has_batch_dim(inputs)
        emissions = ensure_array_has_batch_dim(emissions)
        
        # Run smoothing
        if self.model_type == 'linear':
            posterior = lgssm_smoother(self.params, emissions, inputs)
        else:
            posterior = nlssm_smoother(self.params, emissions, inputs)
        
        return {
            'smoothed_states': posterior.smoothed_means,
            'smoothed_covs': posterior.smoothed_covariances,
            'filtered_states': posterior.filtered_means,
            'filtered_covs': posterior.filtered_covariances,
            'log_likelihood': posterior.marginal_loglik
        }
    
    def predict(self, inputs: jnp.ndarray, initial_state: jnp.ndarray = None) -> Dict[str, jnp.ndarray]:
        """
        Predict future states and emissions.
        
        Args:
            inputs: Input sequence [T, input_dim]
            initial_state: Initial state [state_dim]
            
        Returns:
            Dictionary with predictions
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit_em_algorithm first.")
        
        if initial_state is None:
            initial_state = jnp.zeros(self.ssm.state_dim)
        
        # Ensure inputs have batch dimension
        inputs = ensure_array_has_batch_dim(inputs)
        
        # Predict states
        predicted_states = []
        current_state = initial_state
        
        for t in range(inputs.shape[1]):
            # Predict next state
            if self.model_type == 'linear':
                next_state = (self.params.dynamics.weights @ current_state + 
                            self.params.dynamics.input_weights @ inputs[0, t])
            else:
                next_state = self.ssm.dynamics_fn(current_state, inputs[0, t], self.params)
            
            predicted_states.append(next_state)
            current_state = next_state
        
        predicted_states = jnp.array(predicted_states)
        
        # Predict emissions
        predicted_emissions = []
        for state in predicted_states:
            if self.model_type == 'linear':
                emission = self.params.emissions.weights @ state
            else:
                emission = self.ssm.emission_fn(state, jnp.array([0.0]), self.params)
            predicted_emissions.append(emission)
        
        predicted_emissions = jnp.array(predicted_emissions)
        
        return {
            'predicted_states': predicted_states,
            'predicted_emissions': predicted_emissions
        }


class DynamaxSystemIdentifier:
    """
    System identification using Dynamax state-space models.
    """
    
    def __init__(self, model_type: str = 'linear'):
        """
        Initialize Dynamax system identifier.
        
        Args:
            model_type: Type of model ('linear', 'nonlinear')
        """
        self.model_type = model_type
        self.ssm_model = LoudspeakerStateSpaceModel(model_type)
    
    def identify_linear_parameters(self, u: jnp.ndarray, y: jnp.ndarray,
                                 num_iters: int = 100) -> Dict[str, Any]:
        """
        Identify linear system parameters using Dynamax.
        
        Args:
            u: Input voltage [V]
            y: Output measurements [current, velocity]
            num_iters: Number of EM iterations
            
        Returns:
            Dictionary with identification results
        """
        print("Running linear system identification with Dynamax...")
        
        # Prepare data
        inputs = u.reshape(-1, 1)  # [T, 1]
        emissions = y  # [T, 2]
        
        # Fit model
        fit_results = self.ssm_model.fit_em_algorithm(inputs, emissions, num_iters)
        
        # Get smoothed states
        smooth_results = self.ssm_model.smooth(inputs, emissions)
        
        # Extract parameters
        if self.model_type == 'linear':
            params = {
                'A': self.ssm_model.params.dynamics.weights,
                'B': self.ssm_model.params.dynamics.input_weights,
                'C': self.ssm_model.params.emissions.weights,
                'Q': self.ssm_model.params.dynamics.cov,
                'R': self.ssm_model.params.emissions.cov
            }
        else:
            params = {
                'dynamics_cov': self.ssm_model.params.dynamics.cov,
                'emissions_cov': self.ssm_model.params.emissions.cov
            }
        
        # Make predictions
        predictions = self.ssm_model.predict(inputs)
        
        return {
            'parameters': params,
            'predictions': predictions,
            'smoothed_states': smooth_results['smoothed_states'],
            'fit_results': fit_results,
            'convergence': {
                'method': 'dynamax_linear',
                'final_log_likelihood': fit_results['final_log_likelihood'],
                'iterations': num_iters
            }
        }
    
    def identify_nonlinear_parameters(self, u: jnp.ndarray, y: jnp.ndarray,
                                    num_iters: int = 100) -> Dict[str, Any]:
        """
        Identify nonlinear system parameters using Dynamax.
        
        Args:
            u: Input voltage [V]
            y: Output measurements [current, velocity]
            num_iters: Number of EM iterations
            
        Returns:
            Dictionary with identification results
        """
        print("Running nonlinear system identification with Dynamax...")
        
        # Prepare data
        inputs = u.reshape(-1, 1)  # [T, 1]
        emissions = y  # [T, 2]
        
        # Fit model
        fit_results = self.ssm_model.fit_em_algorithm(inputs, emissions, num_iters)
        
        # Get smoothed states
        smooth_results = self.ssm_model.smooth(inputs, emissions)
        
        # Extract parameters
        params = {
            'dynamics_cov': self.ssm_model.params.dynamics.cov,
            'emissions_cov': self.ssm_model.params.emissions.cov
        }
        
        # Make predictions
        predictions = self.ssm_model.predict(inputs)
        
        return {
            'parameters': params,
            'predictions': predictions,
            'smoothed_states': smooth_results['smoothed_states'],
            'fit_results': fit_results,
            'convergence': {
                'method': 'dynamax_nonlinear',
                'final_log_likelihood': fit_results['final_log_likelihood'],
                'iterations': num_iters
            }
        }


def dynamax_identification_method(u: jnp.ndarray, y: jnp.ndarray,
                                model_type: str = 'linear',
                                **kwargs) -> Dict[str, Any]:
    """
    Dynamax system identification method.
    
    Args:
        u: Input voltage [V]
        y: Output measurements [current, velocity]
        model_type: Type of model ('linear', 'nonlinear')
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with identification results
    """
    print(f"Running Dynamax identification with {model_type} model...")
    
    # Create identifier
    identifier = DynamaxSystemIdentifier(model_type)
    
    # Get parameters
    num_iters = kwargs.get('num_iters', 100)
    
    # Run identification
    if model_type == 'linear':
        result = identifier.identify_linear_parameters(u, y, num_iters)
    else:
        result = identifier.identify_nonlinear_parameters(u, y, num_iters)
    
    return result
