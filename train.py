import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm
import functools
import wandb
import os
import argparse

from models import ScoreNet
from losses import loss_fn
from sdes import simpleSDE, subVPSDE

def main():

    try:
        device = 'cuda'
    except:
        device = 'cpu'
        print('You should turn on your GPU environment.')


    if args.sde == 'simple':
        sigma =  wandb.config.sigma
        sde = simpleSDE(sigma=sigma)
    elif args.sde == 'subvp':
        beta_min, beta_max = 0.1, 20
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max)
    else:
        print('Please provide an existing SDE.')

    marginal_prob_std_fn = functools.partial(sde.marginal_prob_std, sigma=sigma)

    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)

    n_epochs =  wandb.config.epochs
    batch_size =  wandb.config.batch_size
    lr = wandb.config.lr

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = Adam(score_model.parameters(), lr=lr)
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(device)    
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss so far.
        wandb.log({
            'loss': avg_loss / num_items
        })
        pbar.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), f'./checkpoints/ckpt_{avg_loss / num_items}_{epoch}_{sigma}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sde', default='simple')
    args = parser.parse_args()

    dataset = FashionMNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    if os.path.exists('./checkpoints')==False:
      os.mkdir('./checkpoints')
    wandb.login()
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize', 
            'name': 'loss'
            },
        'parameters': {
            'sigma': {'values': [15.0, 25.0, 35.0]},
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'values': [50, 60, 80]},
            'lr': {'max': 0.001, 'min': 0.0001}
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="project-name")
    wandb.agent(sweep_id, function=main, count=6)