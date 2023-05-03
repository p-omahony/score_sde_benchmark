import torch
import numpy as np


class simpleSDE:
  def __init__(self, sigma):
    self.sigma = sigma

  def marginal_prob_std(self, t, device):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.  
    
    Returns:
      The standard deviation.
    """    
    t = torch.tensor(t, device=device)
    return torch.sqrt((self.sigma**(2 * t) - 1.) / 2. / np.log(self.sigma))

  def diffusion_coeff(self, t, device):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
    
    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(self.sigma**t, device=device)
  
class subVPSDE:
  def __init__(self, beta_min=0.1, beta_max=20):
    """Construct the sub-VP SDE that excels at likelihoods.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    self.beta_0 = beta_min
    self.beta_1 = beta_max


  def marginal_prob_std(self, t, device):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.  
    
    Returns:
      The standard deviation.
    """    
    t = torch.tensor(t, device=device)
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    std = 1 - torch.exp(2. * log_mean_coeff)
    return std
  
  def diffusion_coeff(self, t, device):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
    
    Returns:
      The vector of diffusion coefficients.
    """
    t = torch.tensor(t, device=device)
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return diffusion