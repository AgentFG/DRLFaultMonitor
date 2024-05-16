import gymnasium as gym
from stable_baselines3 import SAC, PPO, DQN
import torch
import torch.nn as nn
import sys
sys.path.append('D:\postgraduate\WorkForISSRE\code\TodyNet\src')
from net import GNNStack
from utils import AverageMeter
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

top1 = AverageMeter('Acc', ':6.2f')

# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)

# BipedalWalker-v3
# env = gym.make('BipedalWalker-v3',
#                hardcore=True,
#                render_mode='rgb_array')
# CartPole-v1
# env = gym.make('CartPole-v1')
# InvertedDoublePendulum-v4
# env = gym.make('InvertedDoublePendulum-v4', 
#                render_mode='rgb_array')
# Walker2d-v4
# env = gym.make('Walker2d-v4', 
#             #    reset_noise_scale=0.35,
#                render_mode='rgb_array')
# highway
# config = {
#     "observation": {
#         'type': "Kinematics",
#         'vehicles_count': 5,
#         'features': ['presence', 'x', 'y', 'vx', 'vy', 'cos_h', 'sin_h', 'cos_d', 
#                      'sin_d', 'long_off', 'lat_off', 'ang_off'],
#         'features_range': {
#             'x': [-100, 100],
#             'y': [-100, 100],
#             'vx': [-20, 20],
#             'vy': [-20, 20]
#         },
#         'absolute': False,
#         'order': 'sorted'
#     },
#     'vehicle_density': 3,
#     'duration' : 100
# }

# env = gym.make('highway-fast-v0')
# env.configure(config)
# obs, info = env.reset()
# print(obs)

# LunarLander
env = gym.make('LunarLander-v2')

todeynet =  GNNStack(gnn_model_type='dyGIN2d', num_layers=3, 
            groups=4, pool_ratio=0.2, kern_size=[9,5,3], 
            in_dim=64, hidden_dim=128, out_dim=256, 
            seq_len=20, num_nodes=10, num_classes=2)
todeynet.load_state_dict(torch.load('/home/cy/MTSC/TodyNet/model/03.18_gpu0_dyGIN2d_LunarLander0318_exp.pth'))
todeynet.to('cuda:1')
todeynet.eval()

# model = PPO.load('/home/cy/newGYM/gymnasium_model/model/PPO_CartPole_model_1e4.zip')
# model = DQN.load('/home/cy/newGYM/gymnasium_model/model/exp0104/DQN_LunarLander_model_2e6_0.zip')
# model = SAC.load('/home/cy/newGYM/gymnasium_model/model/exp1130/BipedalWalkerHC-1e6-SAC-0.zip')
# model = PPO.load('/home/cy/newGYM/gymnasium_model/model/PPO-InvertedDoublePendulum-v4.zip')
# model = SAC.load('/home/cy/newGYM/gymnasium_model/model/SAC-Walker2d-5e5.zip') # walker 2d\
# model = PPO.load('/home/cy/newGYM/gymnasium_model/model/PPO_Highway_model_1e5.zip')
model = PPO.load('/home/cy/newGYM/gymnasium_model/model/exp0104/PPO_LunarLander_model_12e5_0.zip')

# labels = torch.zeros(100, dtype=torch.long)
# y_test = []
# for i in range(100):
prob = []
obs, _ = env.reset()
done = False
truncated = False
total_reward = 0
record = deque(maxlen=20)
cnt = 0
while not done and not truncated:
    action, _ = model.predict(obs, deterministic=True)
    state = torch.as_tensor(obs).to(model.device).unsqueeze(0)
    actions = torch.as_tensor(action).to(model.device).unsqueeze(0).unsqueeze(0)
    q_values_pi = model.policy.evaluate_actions(state, actions)[0]
    record.append(torch.cat([state, actions, q_values_pi/100], dim=1))
    obs, reward, done, truncated, info = env.step(action)
    cnt += 1
    if len(record) == 20:
        obs_input = torch.stack(list(record), dim=1)
        # obs_input = torch.as_tensor(np.array(record), dtype=torch.float32).unsqueeze(0)
        obs_input = obs_input.permute(0, 2, 1).unsqueeze(0).float().to('cuda:1')
        a = todeynet(obs_input)
        label = torch.argmax(a, dim=1)
        print(label)
        prob.append(torch.softmax(a, dim=1)[0][1].item())
    total_reward += reward
        # env.render()
env.close()
print(total_reward)
print(cnt)
# if cnt < 500:
#     print("Fail")
# else:
#     print("Success")
    
    # if total_reward < 285:
    #     y_test.append(1)
    # else:
    #     y_test.append(0)
plt.figure()
plt.plot(prob)
plt.savefig('/home/cy/MTSC/TodyNet/figure/LLander/0318/1540/LLander_prob_fail.png')

# y_test = torch.tensor(y_test, dtype=torch.long)
# print(labels)
# print(y_test)
# print(y_test == labels)

# x_test = []
# y_test = []
# for i in range(300):
#     obs, _ = env.reset()
#     done = False
#     truncated = False
#     total_reward = 0
#     obs_record = []

#     while not done and not truncated:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, truncated, info = env.step(action)
#         obs_record.append(obs)
#         total_reward += reward
#     print(total_reward)
#     # it is for cartpole
#     # if total_reward < 500:
#     #     y_test.append(1)
#     # else:
#     #     y_test.append(0)
#     if total_reward <285:
#         y_test.append(1)
#     else:
#         y_test.append(0)
#     x_test.append(obs_record[-30:-10])
    
# x_test = torch.as_tensor(np.array(x_test), dtype=torch.float32).unsqueeze(1)
# x_test = x_test.permute(0, 1, 3, 2)
# y_test = torch.as_tensor(y_test, dtype=torch.long)
# print(x_test.shape, y_test.shape)
# a = todeynet(x_test)
# print(torch.softmax(a, dim=1))
# label = torch.argmax(a, dim=1)
# print(label)
# print(y_test)
# print(label==y_test)
# print((label==y_test).sum())

# X_train = np.load('/home/cy/MTSC/TodyNet/data/UCR/BipedalWalkerHC/X_valid.npy')
# y_train = np.load('/home/cy/MTSC/TodyNet/data/UCR/BipedalWalkerHC/y_valid.npy')
# y_train = torch.as_tensor(y_train, dtype=torch.long)
# # print(X_train.shape, y_train.shape)
# # pred = todeynet(torch.as_tensor(X_train[:10000], dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2))
# # pred = torch.argmax(pred, dim=1)
# # print(pred)
# # print((pred == y_train[:10000]).sum())
# for i in y_train:
#     print(i)

# SAC version
    # state = torch.as_tensor(obs).to(model.device).unsqueeze(0)
    # actions = torch.as_tensor(action).to(model.device).unsqueeze(0)
    # q_values_pi = torch.cat(model.policy.critic(state, actions), dim=1)
    # min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
    # record.append(torch.cat([state, actions, min_qf_pi/100], dim=1))

# PPO 
    # state = torch.as_tensor(obs).to(model.device).unsqueeze(0)
    # actions = torch.as_tensor(action).to(model.device).unsqueeze(0)
    # q_values_pi = model.policy.evaluate_actions(state, actions)[0]
    # record.append(torch.cat([state, actions, q_values_pi/100], dim=1))