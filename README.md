
# Score-Based Generative Modeling Through Stochastic Differential Equations (SDEs) - BENCHMARK

This code allows the realization of a number of experiments which are framed and described in this paper:      

You can see an example of training and sampling in this [notebook](https://colab.research.google.com/drive/1f-pJ5Ogi1wzF9dRCcTA_sbnPg-_PLrMs?usp=sharing).       

### Installation 
`git clone https://github.com/p-omahony/score_sde_benchmark.git`         
`cd score_sde_benchmark`           
`pip install -r requirements.txt`           
### Training
Arguments :      
- `--type`: either `exp` if you want to train with a sweep configuration and with `wandb`, either `normal` if you want to train with a specific configuration
- `--cfg`: path of your specific configuration
- `--sde`: to specify the SDE to use to perturb the data, either `simple` or `subvp`
- `--device`: either `cpu` or `cuda`

### Sampling   
Arguments:        
- `--sampler`: either `ode`, `pc` or `euler`
- `--checkpoints`: the path of the weights of the model (`state_dict`)
- `--sde`: to specify the SDE used to perturb the data, either `simple` or `subvp`
- `--device`: either `cpu` or `cuda`

### References
The code is largely inspired by that of Yang Song in https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing and https://github.com/yang-song/score_sde_pytorch. We completed it to test more parameters and set up the experiment pipeline.
