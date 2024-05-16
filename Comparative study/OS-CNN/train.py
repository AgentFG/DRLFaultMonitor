import os
import torch
from os.path import dirname
from sklearn.metrics import accuracy_score
from util import TSC_multivariate_data_loader, OS_CNN_easy_use

dataset_name_list = [
'BipedalWalkerHCAC',
# 'HopperAC',
# 'HumanoidAC',
# 'Walker2dAC',
# 'InvertedDoublePendulumAC',
]


Result_log_folder = './result/'
dataset_path = dirname("./data/")


for dataset_name in dataset_name_list:
    print('running at:', dataset_name)   
    # load data
    X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(dataset_path, dataset_name)
    print(X_train.shape)
    model = OS_CNN_easy_use(
        Result_log_folder = Result_log_folder, # the Result_log_folder
        dataset_name = dataset_name,           # dataset_name for creat log under Result_log_folder
        device = "cuda:0",                     # Gpu 
        max_epoch = 100,                       # In our expirement the number is 2000 for keep it same with FCN for the example dataset 500 will be enough
        Max_kernel_size = 89, 
        start_kernel_size = 1,
        paramenter_number_of_layer_list = [8*128, (5*128*256 + 2*256*128)/2], 
        quarter_or_half = 4,
        )
    
    model.fit(X_train, y_train, X_test, y_test)
    
    y_predict = model.predict(X_test)
    
    
    print('correct:',y_test)
    print('predict:',y_predict)
    acc = accuracy_score(y_predict, y_test)
    print(acc)
    