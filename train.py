import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm
import functools
import wandb
import os
import yaml
from yaml import Loader
import argparse

from models import ScoreNet, EMA
from losses import loss_fn
from sdes import simpleSDE, subVPSDE

def load_config(cfg_path):
    with open(cfg_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader)
    return cfg

def main():

    try:
        device = 'cuda'
    except:
        device = 'cpu'
        print('You should turn on your GPU environment.')

    if args.type != 'exp':
        cfg = load_config(args.cfg)
    else:
        run = wandb.init()
        cfg = wandb.config 

    if args.sde == 'simple':
        sigma = float(cfg['sigma'])
        sde = simpleSDE(sigma=sigma)
    elif args.sde == 'subvp':
        beta_min, beta_max = float(cfg['beta_min']), float(cfg['beta_max'])
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max)
    else:
        print('Please provide an existing SDE.')

    marginal_prob_std_fn = functools.partial(sde.marginal_prob_std, device=device)

    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)

    if args.ema == 'true':
        ema = EMA(0.999)
        for name, param in score_model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)

    n_epochs =  int(cfg['epochs'])
    batch_size =  int(cfg['batch_size'])
    lr = float(cfg['lr'])       

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
            if args.ema == 'true':
                ema = EMA(0.999)
                for name, param in score_model.named_parameters():
                    if param.requires_grad:
                        ema(name, param.data)
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss so far.
        if args.type =='exp':
            wandb.log({
                'loss': avg_loss / num_items
            })
        pbar.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), f'./checkpoints/ckpt.pth') #_{avg_loss / num_items}_{epoch}_{sigma}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sde', default='simple')
    parser.add_argument('-t', '--type', default='cfg')
    parser.add_argument('-c', '--cfg', default='./configs/simplesde_cfg.yaml')
    parser.add_argument('-e', '--ema', default='false')
    args = parser.parse_args()

    dataset = FashionMNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    if os.path.exists('./checkpoints')==False:
      os.mkdir('./checkpoints')
    if args.type =='exp':
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
    else:
        main()