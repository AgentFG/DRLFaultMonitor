import argparse
from collections import deque
from utils import prepare_agent, transform_input
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time

from src.model import MLSTMfcn
from src.utils import validation, load_datasets
from src.constants import NUM_CLASSES, MAX_SEQ_LEN, NUM_FEATURES


parser = argparse.ArgumentParser(description='DRL Analysis based on TodyNet')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BipedalWalkerHC')
parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')
parser.add_argument('--groups', type=int, default=4, help='the number of time series groups (num_graphs)')
parser.add_argument('--pool_ratio', type=float, default=0.2, help='the ratio of pooling for nodes')
parser.add_argument('--kern_size', type=str, default="9,5,3", help='list of time conv kernel size for each layer')
parser.add_argument('--in_dim', type=int, default=64, help='input dimensions of GNN stacks')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimensions of GNN stacks')
parser.add_argument('--out_dim', type=int, default=256, help='output dimensions of GNN stacks')
parser.add_argument('--smote', dest='smote', action='store_true',
                    help='use smote to balance the data')
args = parser.parse_args()

model_dir = 'model/' + args.dataset + '.pth'

result_save_dir = 'result/' + args.dataset + '.csv'

if args.dataset[-2:] == "AC":
    input_tag = True
    args.dataset = args.dataset[:-2]
else:
    input_tag = False

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

env, model, num_nodes, alg_tag = prepare_agent(args.dataset)
print(num_nodes)

args.kern_size = [ int(l) for l in args.kern_size.split(",") ]


df = pd.DataFrame(columns=['Episode', 'Reward', 'Pre', 'True', 'Probabilities', 'Steps'])
    

dataset = args.dataset
assert dataset in NUM_CLASSES.keys()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))

mlstml = MLSTMfcn(num_classes=NUM_CLASSES[dataset], 
                            max_seq_len=MAX_SEQ_LEN[dataset], 
                            num_features=NUM_FEATURES[dataset])
mlstml.load_state_dict(torch.load('model/'+args.dataset + '.pt'))
mlstml.eval()
mlstml.to(device)

# X_train = torch.load('/home/cy/WorkForISSRE/RQ2/OS-CNN/data/BipedalWalkerHCAC/X_train.pt')
# y_train = torch.load('/home/cy/WorkForISSRE/RQ2/OS-CNN/data/BipedalWalkerHCAC/y_train.pt')


# X_train = X_train.squeeze(1)
# X_train = torch.transpose(X_train, 1, 2)
# print(X_train.shape)
# batch_size = X_train.shape[0]
# seql = X_train.shape[1]
# seq_lens = torch.tensor([seql] * batch_size, dtype=torch.float)
# a = torch.argmax(mlstml(X_train, seq_lens),dim=1)
# print(torch.sum(a == y_train))

seq_length = MAX_SEQ_LEN[dataset]

check_episode = 1

pre_label = np.zeros(check_episode)
true_label = np.zeros(check_episode)

# # for i in range(100):
for i in range(check_episode):
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    record = deque(maxlen=seq_length)
    cnt = 0
    prob = -1
    steps = 0
    ti = []
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        record = transform_input(obs, action, model, record, alg_tag)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        cnt += 1
        if len(record) == seq_length:
            t1 =time.time()
            obs_input = torch.stack(list(record), dim=1)
            obs_input = obs_input.to('cuda:0').to(torch.float32)
            seq_len = torch.tensor([seq_length]).to('cpu')
            a = mlstml(obs_input, seq_len)
            t = time.time()-t1
            print(t)
            ti.append(t)
            label = torch.argmax(a, dim=1)
            if label.item() == 1 and steps==0:
                pre_label[i] = 1
                prob = torch.softmax(a, dim=1)[0][1].item()
                steps = cnt
                
    # if cnt == 1000:
    #     pre_label[i] = 0
    #     continue         
    if cnt < 1000:
        true_label[i] = 1
    # if reward == -100 or cnt == 1000:
    #     true_label[i] = 1
    # if total_reward < 285:
    #     true_label[i] = 1
    print('t', sum(ti)/len(ti))
    print("Episode: ", i, "Reward: ", total_reward, "Pre: ", pre_label[i], "True: ", true_label[i], "Prob: ", prob, 'Steps: ', cnt - steps)
    
    df.loc[len(df)] = [i, total_reward, pre_label[i], true_label[i], prob, cnt - steps]
    

# df.to_csv(result_save_dir, index=False)

    

