import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import functools
import argparse

from sdes import simpleSDE, subVPSDE
from models import ScoreNet
from samplers import ode_sampler


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sde', default='simple')
args = parser.parse_args()

device = 'cpu'

if args.sde == 'simple':
    sigma =  25.0
    sde = simpleSDE(sigma=sigma)
elif args.sde == 'subvp':
    beta_min, beta_max = 0.1, 20
    sde = subVPSDE(beta_min=beta_min, beta_max=beta_max)
else:
    print('Please provide an existing SDE.')

marginal_prob_std_fn = functools.partial(sde.marginal_prob_std, device=device)
diffusion_coeff_fn = functools.partial(sde.diffusion_coeff, device=device)

ckpt = torch.load('./checkpoints/best_so_far.pth', map_location=device)
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model.load_state_dict(ckpt)
score_model = score_model.to(device)

sample_batch_size = 64 
sampler = ode_sampler

samples = sampler(score_model, 
                  marginal_prob_std_fn,
                  diffusion_coeff_fn, 
                  sample_batch_size, 
                  atol=1e-5,
                  rtol=1e-5,
                  eps=1e-3,
                  device=device)

samples = samples.clamp(0.0, 1.0)
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()