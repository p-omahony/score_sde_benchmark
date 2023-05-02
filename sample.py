import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import functools

from sde import marginal_prob_std, diffusion_coeff
from models import ScoreNet
from samplers import ode_sampler


device = 'cuda'
ckpt = torch.load('./checkpoints/ckpt.pth', map_location=device)
sigma =  25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model.load_state_dict(ckpt)
score_model = score_model.to(device)

sample_batch_size = 64 
sampler = ode_sampler

samples = sampler(score_model, 
                  marginal_prob_std_fn,
                  diffusion_coeff_fn, 
                  sample_batch_size, 
                  device=device)

samples = samples.clamp(0.0, 1.0)
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()