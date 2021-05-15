# Recommendation System using Reinforcement Learning with Generative Adversarial Training

A pytorch implementation of *Model-Based Reinforcement Learning with Adversarial Training for Online Recommendation* (https://arxiv.org/pdf/1911.03845.pdf).

## Dependency
 - Pytorch version: 1.1.0
 
## Usage: 
In the directory of IRecGAN, type command: 

```
python main.py --click ../simulation_task1/gen_click.txt --reward ../simulation_task1/gen_reward.txt --action ../simulation_task1/gen_action.txt --model LSTM --nhid 128 --n_layers_usr 2 --optim_nll adam --optim_adv adam --batch_size 128
```

The variable **interact** in main.py enables online training and evaluation with the environment in ./simulation_task1. However, many routes in ./simulation_task1 have to be changed.  

**./simulation_task1** contains a simulated environment(different from paper) and offline data can be generated by: 

python Generate_data.py

