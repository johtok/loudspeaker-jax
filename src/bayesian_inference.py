"""
Bayesian Inference for Loudspeaker System Identification.

This module implements Bayesian parameter estimation using NumPyro and BlackJAX
for uncertainty quantification in loudspeaker model parameters.

Author: Research Team
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
import blackjax
from blackjax import nuts, window_adaptation

from nonlinear_loudspeaker_model import NonlinearLoudspeakerModel


class BayesianLoudspeakerInference:
    """
    Bayesian inference for loudspeaker system identification.
    
    Implements parameter uncertainty quantification using:
    - NumPyro for probabilistic programming
    - BlackJAX for advanced MCMC sampling (NUTS, HMC)
    """
    
    def __init__(self, model: NonlinearLoudspeakerModel):
        """Initialize Bayesian inference with model."""
        self.model = model
        self.prior_params = None
        self.posterior_samples = None
    
    def set_priors(self, prior_params: Dict[str, Dict[str, float]]):
        """
        Set prior distributions for model parameters.
        
        Args:
            prior_params: Dictionary with prior parameters for each model parameter
                         Format: {'param_name': {'mean': float, 'std': float}}
        """
        self.prior_params = prior_params
    
    def default_priors(self) -> Dict[str, Dict[str, float]]:
        """Set default prior distributions based on typical loudspeaker parameters."""
        return {
            'Re': {'mean': 6.8, 'std': 1.0},
            'Le': {'mean': 0.5e-3, 'std': 0.1e-3},
            'Bl': {'mean': 3.2, 'std': 0.5},
            'M': {'mean': 12e-3, 'std': 2e-3},
            'K': {'mean': 1200, 'std': 200},
            'Rm': {'mean': 0.8, 'std': 0.2},
            'L20': {'mean': 0.1e-3, 'std': 0.02e-3},
            'R20': {'mean': 0.5, 'std': 0.1},
            'Bl_nl_0': {'mean': 0.0, 'std': 10.0},
            'Bl_nl_1': {'mean': 0.0, 'std': 10.0},
            'Bl_nl_2': {'mean': -50.0, 'std': 20.0},
            'Bl_nl_3': {'mean': -0.1, 'std': 0.05},
            'K_nl_0': {'mean': 0.0, 'std': 20.0},
            'K_nl_1': {'mean': 0.0, 'std': 20.0},
            'K_nl_2': {'mean': 100.0, 'std': 50.0},
            'K_nl_3': {'mean': 0.0, 'std': 10.0}
        }
    
    def model_numpyro(self, u: jnp.ndarray, y_measured: jnp.ndarray, 
                     x0: jnp.ndarray = None) -> None:
        """
        Define NumPyro probabilistic model.
        
        Args:
            u: Input voltage [V]
            y_measured: Measured outputs [current, velocity]
            x0: Initial state
        """
        # Set priors if not already set
        if self.prior_params is None:
            self.prior_params = self.default_priors()
        
        # Sample linear parameters
        Re = numpyro.sample("Re", dist.Normal(self.prior_params['Re']['mean'], 
                                            self.prior_params['Re']['std']))
        Le = numpyro.sample("Le", dist.Normal(self.prior_params['Le']['mean'], 
                                            self.prior_params['Le']['std']))
        Bl = numpyro.sample("Bl", dist.Normal(self.prior_params['Bl']['mean'], 
                                            self.prior_params['Bl']['std']))
        M = numpyro.sample("M", dist.Normal(self.prior_params['M']['mean'], 
                                          self.prior_params['M']['std']))
        K = numpyro.sample("K", dist.Normal(self.prior_params['K']['mean'], 
                                          self.prior_params['K']['std']))
        Rm = numpyro.sample("Rm", dist.Normal(self.prior_params['Rm']['mean'], 
                                            self.prior_params['Rm']['std']))
        L20 = numpyro.sample("L20", dist.Normal(self.prior_params['L20']['mean'], 
                                              self.prior_params['L20']['std']))
        R20 = numpyro.sample("R20", dist.Normal(self.prior_params['R20']['mean'], 
                                              self.prior_params['R20']['std']))
        
        # Sample nonlinear parameters
        Bl_nl = jnp.array([
            numpyro.sample("Bl_nl_0", dist.Normal(self.prior_params['Bl_nl_0']['mean'], 
                                                 self.prior_params['Bl_nl_0']['std'])),
            numpyro.sample("Bl_nl_1", dist.Normal(self.prior_params['Bl_nl_1']['mean'], 
                                                 self.prior_params['Bl_nl_1']['std'])),
            numpyro.sample("Bl_nl_2", dist.Normal(self.prior_params['Bl_nl_2']['mean'], 
                                                 self.prior_params['Bl_nl_2']['std'])),
            numpyro.sample("Bl_nl_3", dist.Normal(self.prior_params['Bl_nl_3']['mean'], 
                                                 self.prior_params['Bl_nl_3']['std']))
        ])
        
        K_nl = jnp.array([
            numpyro.sample("K_nl_0", dist.Normal(self.prior_params['K_nl_0']['mean'], 
                                                self.prior_params['K_nl_0']['std'])),
            numpyro.sample("K_nl_1", dist.Normal(self.prior_params['K_nl_1']['mean'], 
                                                self.prior_params['K_nl_1']['std'])),
            numpyro.sample("K_nl_2", dist.Normal(self.prior_params['K_nl_2']['mean'], 
                                                self.prior_params['K_nl_2']['std'])),
            numpyro.sample("K_nl_3", dist.Normal(self.prior_params['K_nl_3']['mean'], 
                                                self.prior_params['K_nl_3']['std']))
        ])
        
        # Set model parameters
        params = {
            'Re': Re, 'Le': Le, 'Bl': Bl, 'M': M, 'K': K, 'Rm': Rm,
            'L20': L20, 'R20': R20, 'Bl_nl': Bl_nl, 'K_nl': K_nl,
            'L_nl': jnp.zeros(4), 'Li_nl': jnp.zeros(4)
        }
        self.model.set_parameters(params)
        
        # Predict outputs
        y_pred = self.model.predict(u, x0)
        
        # Likelihood
        noise_std = numpyro.sample("noise_std", dist.HalfNormal(0.1))
        numpyro.sample("y", dist.Normal(y_pred, noise_std), obs=y_measured)
    
    def run_mcmc_numpyro(self, u: jnp.ndarray, y_measured: jnp.ndarray,
                        x0: jnp.ndarray = None, num_samples: int = 1000,
                        num_warmup: int = 500, num_chains: int = 4) -> Dict[str, Any]:
        """
        Run MCMC sampling using NumPyro.
        
        Args:
            u: Input voltage [V]
            y_measured: Measured outputs [current, velocity]
            x0: Initial state
            num_samples: Number of posterior samples
            num_warmup: Number of warmup samples
            num_chains: Number of chains
            
        Returns:
            Dictionary with MCMC results
        """
        print("Running MCMC with NumPyro...")
        
        # Create NUTS kernel
        kernel = NUTS(self.model_numpyro)
        
        # Run MCMC
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, 
                   num_chains=num_chains)
        mcmc.run(jax.random.PRNGKey(42), u, y_measured, x0)
        
        # Get samples
        samples = mcmc.get_samples()
        
        # Calculate posterior statistics
        posterior_stats = {}
        for param_name, param_samples in samples.items():
            if param_name != 'y':  # Skip observed data
                posterior_stats[param_name] = {
                    'mean': jnp.mean(param_samples),
                    'std': jnp.std(param_samples),
                    'samples': param_samples
                }
        
        return {
            'samples': samples,
            'posterior_stats': posterior_stats,
            'mcmc': mcmc
        }
    
    def run_mcmc_blackjax(self, u: jnp.ndarray, y_measured: jnp.ndarray,
                         x0: jnp.ndarray = None, num_samples: int = 1000,
                         num_warmup: int = 500) -> Dict[str, Any]:
        """
        Run MCMC sampling using BlackJAX.
        
        Args:
            u: Input voltage [V]
            y_measured: Measured outputs [current, velocity]
            x0: Initial state
            num_samples: Number of posterior samples
            num_warmup: Number of warmup samples
            
        Returns:
            Dictionary with MCMC results
        """
        print("Running MCMC with BlackJAX...")
        
        # Define log-posterior function
        def log_posterior(params):
            # Set model parameters
            self.model.set_parameters(params)
            
            # Predict outputs
            y_pred = self.model.predict(u, x0)
            
            # Calculate log-likelihood
            log_likelihood = jnp.sum(dist.Normal(y_pred, 0.1).log_prob(y_measured))
            
            # Calculate log-prior (assuming independent normal priors)
            log_prior = 0.0
            for param_name, param_value in params.items():
                if param_name in self.prior_params:
                    prior_mean = self.prior_params[param_name]['mean']
                    prior_std = self.prior_params[param_name]['std']
                    log_prior += dist.Normal(prior_mean, prior_std).log_prob(param_value)
            
            return log_likelihood + log_prior
        
        # Initialize NUTS sampler
        rng_key = jax.random.PRNGKey(42)
        
        # Create initial parameters
        initial_params = {}
        for param_name, prior_info in self.prior_params.items():
            initial_params[param_name] = prior_info['mean']
        
        # Run window adaptation
        warmup = window_adaptation(nuts, log_posterior)
        state, kernel, _ = warmup.run(rng_key, initial_params, num_warmup)
        
        # Run sampling
        def inference_loop(rng_key, initial_state, kernel, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, info = kernel(rng_key, state)
                return state, (state, info)
            
            keys = jax.random.split(rng_key, num_samples)
            final_state, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
            return final_state, states, infos
        
        final_state, states, infos = inference_loop(rng_key, state, kernel, num_samples)
        
        # Extract samples
        samples = {}
        for param_name in initial_params.keys():
            samples[param_name] = states.position[param_name]
        
        # Calculate posterior statistics
        posterior_stats = {}
        for param_name, param_samples in samples.items():
            posterior_stats[param_name] = {
                'mean': jnp.mean(param_samples),
                'std': jnp.std(param_samples),
                'samples': param_samples
            }
        
        return {
            'samples': samples,
            'posterior_stats': posterior_stats,
            'final_state': final_state,
            'infos': infos
        }
    
    def predict_with_uncertainty(self, u: jnp.ndarray, x0: jnp.ndarray = None,
                                num_samples: int = 100) -> Dict[str, jnp.ndarray]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            u: Input voltage [V]
            x0: Initial state
            num_samples: Number of posterior samples to use
            
        Returns:
            Dictionary with prediction statistics
        """
        if self.posterior_samples is None:
            raise ValueError("No posterior samples available. Run MCMC first.")
        
        # Sample parameters from posterior
        predictions = []
        for i in range(num_samples):
            # Sample parameters
            params = {}
            for param_name, param_samples in self.posterior_samples.items():
                if param_name != 'y':  # Skip observed data
                    idx = jax.random.randint(jax.random.PRNGKey(i), (), 0, len(param_samples))
                    params[param_name] = param_samples[idx]
            
            # Set model parameters and predict
            self.model.set_parameters(params)
            y_pred = self.model.predict(u, x0)
            predictions.append(y_pred)
        
        predictions = jnp.array(predictions)
        
        return {
            'mean': jnp.mean(predictions, axis=0),
            'std': jnp.std(predictions, axis=0),
            'samples': predictions
        }


def bayesian_identification_method(u: jnp.ndarray, y: jnp.ndarray,
                                 method: str = 'numpyro',
                                 **kwargs) -> Dict[str, Any]:
    """
    Bayesian system identification method.
    
    Args:
        u: Input voltage [V]
        y: Output measurements [current, velocity]
        method: MCMC method ('numpyro', 'blackjax')
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with identification results
    """
    print(f"Running Bayesian identification using {method}...")
    
    # Create model
    model = NonlinearLoudspeakerModel()
    
    # Create Bayesian inference
    bayesian_inference = BayesianLoudspeakerInference(model)
    
    # Set priors
    prior_params = kwargs.get('prior_params', bayesian_inference.default_priors())
    bayesian_inference.set_priors(prior_params)
    
    # Get MCMC parameters
    num_samples = kwargs.get('num_samples', 1000)
    num_warmup = kwargs.get('num_warmup', 500)
    num_chains = kwargs.get('num_chains', 4)
    
    # Run MCMC
    if method == 'numpyro':
        result = bayesian_inference.run_mcmc_numpyro(
            u, y, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains
        )
    elif method == 'blackjax':
        result = bayesian_inference.run_mcmc_blackjax(
            u, y, num_samples=num_samples, num_warmup=num_warmup
        )
    else:
        raise ValueError(f"Unknown MCMC method: {method}")
    
    # Store posterior samples
    bayesian_inference.posterior_samples = result['samples']
    
    # Get posterior mean parameters
    posterior_params = {}
    for param_name, stats in result['posterior_stats'].items():
        posterior_params[param_name] = stats['mean']
    
    # Generate predictions with posterior mean
    model.set_parameters(posterior_params)
    predictions = model.predict(u)
    
    return {
        'model': model,
        'parameters': posterior_params,
        'predictions': predictions,
        'posterior_stats': result['posterior_stats'],
        'bayesian_inference': bayesian_inference,
        'convergence': {
            'method': method,
            'num_samples': num_samples,
            'num_warmup': num_warmup
        }
    }
