import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from DeepRL_DTA import *

class DeepRL_DTA_model_GIN(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(DeepRL_DTA_model_GIN, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        self.molecule_encoder = Transformer_lar()

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        smile = data.smile
        # print("=======11111==========",batch)
        # print("=======22222==========",target.shape)
        # print("=======33333==========",smile.shape)
        # exit()

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x_sub = F.dropout(x, p=0.2, training=self.training)  ##obtain drug graph


        ####protein sequence embedding
        embedded_xt_sub = self.embedding_xt(target)  ###embedding protein sequence
        embedded_xt_llm = self.molecule_encoder(target)  #### embedding protein sequence by llm
        beta = 0.1  ##0.3
        embedded_xt = beta*embedded_xt_sub + (1-beta)*embedded_xt_llm
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # print("=======target=========",target)
        # print("=======smile=========",smile)
        # exit()

        # print("=================",embedded_xt_sub.shape)
        # print("=================",embedded_xt_llm.shape)
        # print("=================",conv_xt.shape)
        # exit()

        ###drug smiles embedding
        # embedded_xd_sub = self.embedding_xt(smile)  ###embedding drug smiles 
        embedded_xd_llm = self.molecule_encoder(smile)  #### embedding drug smiles by llm       
        # beta = 0.5
        # embedded_xd = beta*embedded_xd_sub + (1-beta)*embedded_xd_llm
        conv_xd = self.conv_xt_1(embedded_xd_llm)
        # # flatten
        xd = conv_xd.view(-1, 32 * 121)
        xd = self.fc1_xt(xd)
        alpha = 0.1 ##0.3-->ci=0.897568 mes=0.22174
        x = alpha*x_sub + (1-alpha)*xd  ###drug_graph+drug_smiles
        # print("========1111=========",conv_xd.shape)
        # print("========2222=========",xd.shape)
        # print("========3333=========",xt.shape)
        # exit()

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
