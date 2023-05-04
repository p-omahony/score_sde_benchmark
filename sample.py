import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
import functools
import argparse

from sdes import simpleSDE, subVPSDE
from models import ScoreNet
from samplers import ode_sampler, Euler_Maruyama_sampler, pc_sampler


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--sde', default='simple')
parser.add_argument('-s', '--sampler', default='ode')
parser.add_argument('-c', '--checkpoints', default='./checkpoints/best_simple_wo_ema.pth')
parser.add_argument('-d', '--device', default='cpu')
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

marginal_prob_std_fn = functools.partial(sde.marginal_prob_std, device=args.device)
diffusion_coeff_fn = functools.partial(sde.diffusion_coeff, device=args.device)

ckpt = torch.load(args.checkpoints, map_location=device)
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model.load_state_dict(ckpt)
score_model = score_model.to(args.device)

sample_batch_size = 128 

if args.sampler == 'ode':
    sampler = ode_sampler
    samples = sampler(score_model, 
                    marginal_prob_std_fn,
                    diffusion_coeff_fn, 
                    sample_batch_size, 
                    atol=1e-5,
                    rtol=1e-5,
                    eps=1e-3,
                    device=args.device)
elif args.sampler == 'euler':
    sampler = Euler_Maruyama_sampler
    sampler(score_model, 
                  marginal_prob_std_fn,
                  diffusion_coeff_fn, 
                  500,
                  1e-3,
                  sample_batch_size, 
                  device=device)
elif args.sampler == 'pc':
    sampler = Euler_Maruyama_sampler
    sampler(score_model, 
                  marginal_prob_std_fn,
                  diffusion_coeff_fn, 
                  sample_batch_size,
                  500,
                  snr=0.16,
                  eps=1e-3, 
                  device=args.device)

samples = samples.clamp(0.0, 1.0)
c=0
for sample in samples:
    save_image(sample, f'./samples/generated/{c}.png')
    c+=1
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()