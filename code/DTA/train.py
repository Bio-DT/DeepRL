import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.deeprlgatgcn import DeepRL_DTA_model_GAT
from models.deeprlginconv import DeepRL_DTA_model_GIN
from utils import *


'''
python /media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/DTA/train.py 0 0 0  ---第一个参数:数据-davis   第二个参数:选模型   第三个参数:GPU
python /media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/DTA/train.py 1 1 0  ---第一个参数:数据-kiba   第二个参数:选模型   第三个参数:GPU

'''


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    
    total_preds = total_preds.squeeze().cpu().detach().numpy()
    total_labels = total_labels.squeeze().cpu().detach().numpy()
    # print("========",total_labels)

    return total_labels, total_preds


datasets = [['davis','kiba'][int(sys.argv[1])]] 
modeling = [DeepRL_DTA_model_GIN, DeepRL_DTA_model_GAT][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
print('cuda_name:', cuda_name)

# # 选择 GPU 或 CPU 作为设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRAIN_BATCH_SIZE = 64  
TEST_BATCH_SIZE = 64 
LR = 0.00005  #0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

save_model = '/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/DTA/save_model'
save_result = '/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/DTA/save_result'

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = '/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/DTA/data/processed/' + dataset + '_train.pt'
    processed_data_file_test = '/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/DTA/data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/DTA/data', dataset=dataset+'_train')
        test_data = TestbedDataset(root='/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/DTA/data', dataset=dataset+'_test')

        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)

       # ##load model
        # load_model_path = os.path.join(save_model, 'model_' + model_st + '_' + dataset +  '.pt')
        # if os.path.exists(load_model_path):
        #     save_file_dict = torch.load(load_model_path)
        #     # 处理参数名称（如果保存时使用了多GPU训练，参数名称可能带有 "module." 前缀）
        #     new_save_file_dict = {k.replace("module.", ""): v for k, v in save_file_dict.items()}
            
        #     # 获取当前模型的 state_dict
        #     model_state_dict = model.state_dict()
            
        #     # 只加载匹配的参数
        #     for key, value in new_save_file_dict.items():
        #         if key in model_state_dict and model_state_dict[key].shape == value.shape:
        #             model_state_dict[key] = value
        #         else:
        #             print(f"Skipping {key} due to size mismatch or missing key.")
            
        #     # 加载调整后的参数
        #     model.load_state_dict(model_state_dict, strict=False)
        #     print(f"Model parameters loaded from {load_model_path}")
        # else:
        #     print(f"No saved model found at {load_model_path}, training from scratch.")


        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_ci = 0
        best_epoch = -1
        
        model_file_name = 'model_' + model_st + '_' + dataset + '.pt'  ##model_file_name = 'model_' + model_st + '_' + dataset +  '.pt'
        name = os.path.join(save_model, model_file_name)
        result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        result = os.path.join(save_result, result_file_name)

        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch+1)
            G, P = predicting(model, device, test_loader)   #G=label P=pre

            # ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
            ret = [rmse(G,P),mse(G,P),r2(G, P),spearman(G,P),ci(G,P)]


            if ret[-1]>best_ci:  ##ret[1]<best_mse:  ret[-1]>best_ci
                torch.save(model.state_dict(), name)   ###保存模型
                with open(result,'w') as f:
                    f.write(','.join(map(str,ret)))
                best_epoch = epoch+1
                best_mse = ret[1]
                best_ci = ret[-1]
                # best_ci = ret[-1]
                # best_mse = ret[1]
                best_r2 = ret[2]
                print('ci improved at epoch ', best_epoch, '; best_mse,best_ci,best_r2:', best_mse,best_ci,best_r2,model_st,dataset)
            else:
                print(ret[-1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci,best_r2:', best_mse,best_ci,best_r2,model_st,dataset)
                print('current_ci:', ret[-1],model_st,dataset)
                print('current_mse:', ret[1],model_st,dataset)
                print('current_r2:', ret[2],model_st,dataset)

