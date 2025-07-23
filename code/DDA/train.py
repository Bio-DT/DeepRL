import timeit
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
from preprocessing import *
from model.DeepRL_DDA import DeepRL_DDA_model
from metric import *

"""install env
pytorch--> pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
dgl--> pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html (linux);  pip install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html (windows)
network--> pip install networkx
pip install -U scikit-learn
pip install fairscale
"""


device = torch.device('cuda')

# print(
#     "Number of Parameters: ",
#     sum(p.numel() for p in model.parameters() if p.requires_grad),  #numel() 计算张量中的元素个数
#     flush=True,
# )



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=10, help='iteration')  #10
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train') #200
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate') #2e-4-->不错
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay') #1e-3
    parser.add_argument('--random_seed', type=int, default=666, help='random seed')  #666
    parser.add_argument('--neighbor', type=int, default=5, help='neighbor') #5-->不错
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default='0.5', type=float, help='dropout')  #0.4-->不错
    parser.add_argument('--gt_layer', default='2', type=int, help='graph transformer layer') #2
    parser.add_argument('--gt_head', default='2', type=int, help='graph transformer head') #2
    parser.add_argument('--gt_out_dim', default='256', type=int, help='graph transformer output dimension')#200
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer') #2                       
    parser.add_argument('--hgt_head', default='8', type=int, help='heterogeneous graph transformer head') #8
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension') #64
    parser.add_argument('--hgt_head_dim', default='32', type=int, help='heterogeneous graph transformer head dimension') #25
    parser.add_argument('--hgt_out_dim', default='256', type=int, help='heterogeneous graph transformer output dimension')#200
    parser.add_argument('--tr_layer', default='2', type=int, help='transformer layer') #2
    parser.add_argument('--tr_head', default='4', type=int, help='transformer head') #4

    args = parser.parse_args()
    args.data_dir = '/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/DDA/data/' + args.dataset + '/'
    args.result_dir = 'Result/' + args.dataset + '/AMNTDDA/'

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = data_part(data, args)

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)

    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)
    all_sample = torch.tensor(data['all_drdi']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs = [], []

    print('Dataset:', args.dataset)
    best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0

    for i in range(args.iteration): 

        model = DeepRL_DDA_model(args)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        X_train = torch.LongTensor(data['X_train'][i]).to(device) 
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)

        for epoch in range(args.epochs):
            _, train_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_train)
            train_loss = cross_entropy(train_score, torch.flatten(Y_train))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                dr_representation, test_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_test)

            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()

            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                       round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            # print('\t\t'.join(map(str, show)))

            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                print('AUC improved','\tbest_auc:', best_auc, ';\tbest_aupr:', best_aupr)

            else:
                print('AUC not improved', 'best_auc:', best_auc, ';\tbest_aupr:', best_aupr)   




