import torch
import gymnasium as gym
from stable_baselines3 import SAC, PPO


def prepare_agent(env_name, input_tag=False):
    if env_name == 'LunarLander':
        env = gym.make('LunarLander-v2')
        model = PPO.load('gymmodel/LunarLander-v2-PPO.zip')
        if input_tag:
            num_nodes = 10
        else:
            num_nodes = 8
        
        alg_tag = 'PPO'
    elif env_name == 'CartPole':
        env = gym.make('CartPole-v1')
        model = PPO.load('gymmodel/CartPole-v1-PPO.zip')
        if input_tag:
            num_nodes = 6
        else:
            num_nodes = 4
        
        alg_tag = 'PPO'
          
    elif env_name == 'BipedalWalkerHC':
        env = gym.make('BipedalWalker-v3',
                       hardcore=True,)
        model = SAC.load('gymmodel/BipedalWalker-v3-hardcore-SAC.zip')
        if input_tag:
            num_nodes = 28
        else:
            num_nodes = 24
        
        alg_tag = 'SAC'

    elif env_name == 'Walker2d':
        env = gym.make('Walker2d-v4')
        model = SAC.load('gymmodel/Walker2d-v4-SAC.zip')
        if input_tag:
            num_nodes = 23
        else:
            num_nodes = 17
        
        alg_tag = 'SAC'
    elif env_name == 'InvertedDoublePendulum':
        env = gym.make('InvertedDoublePendulum-v4')
        model = PPO.load('gymmodel/InvertedDoublePendulum-v4-PPO.zip')
        if input_tag:
            num_nodes = 12
        else:
            num_nodes = 11
        
        alg_tag = 'PPO'
    
    elif env_name == 'Hopper':
        env = gym.make('Hopper-v4')
        model = SAC.load('gymmodel/Hopper-v4-SAC.zip')
        if input_tag:
            num_nodes = 14
        else:
            num_nodes = 11
    
        alg_tag = 'SAC'
    
    elif env_name == 'Ant':
        env = gym.make('Ant-v4')
        model = SAC.load('gymmodel/Ant-v4-SAC.zip')
        if input_tag:
            num_nodes = 36
        else:
            num_nodes = 27
        
        alg_tag = 'SAC'
    
    elif env_name == 'Humanoid':
        env = gym.make('Humanoid-v4')
        model = SAC.load('gymmodel/Humanoid-v4-SAC.zip')
        if input_tag:
            num_nodes = 62
        else:
            num_nodes = 45
        
        alg_tag = 'SAC'
        
    return env, model, num_nodes, alg_tag

def transform_input(obs, action, model, record, alg_tag):
    # SAC version
    state = torch.as_tensor(obs).to(model.device).unsqueeze(0)
    actions = torch.as_tensor(action).to(model.device).unsqueeze(0)
    record.append(torch.cat([state[:,:45], actions], dim=1))

    return record
