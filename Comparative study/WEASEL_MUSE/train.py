from pyts.multivariate.transformation import WEASELMUSE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import pickle

# 生成样本数据
dataset = 'Walker2dAC'
path = 'data/' + dataset + '/'
model_save_dir = 'model/' + dataset + '/'

X_train = torch.load(path + '/X_train.pt').squeeze(1).detach().cpu().numpy()
y_train = torch.load(path + '/y_train.pt').detach().cpu().numpy()
X_test = torch.load(path + '/X_valid.pt').squeeze(1).detach().cpu().numpy()
y_test = torch.load(path + '/y_valid.pt').detach().cpu().numpy()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# 实例化WEASELMUSE对象
weaselmuse = WEASELMUSE(word_size=4, n_bins=2, window_sizes=[5, 10],
                        chi2_threshold=15, sparse=False, strategy='uniform')

# 将训练集转换为词频序列
weaselmuse.fit(X_train, y_train)
with open(model_save_dir + 'weaselmuse.pkl', 'wb') as f:
    pickle.dump(weaselmuse, f)
X_train_weaselmuse = weaselmuse.transform(X_train)

# 将测试集转换为词频序列
X_test_weaselmuse = weaselmuse.transform(X_test)

# 实例化随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 在训练集上训练分类器
clf.fit(X_train_weaselmuse, y_train)

with open(model_save_dir + 'RandomForest.pkl', 'wb') as f:
    pickle.dump(clf, f)

# 在测试集上评估分类器
score = clf.score(X_test_weaselmuse, y_test)

print("Classification accuracy: ", score)