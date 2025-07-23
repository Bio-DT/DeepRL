import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

from layers import EGCL, ConstrainedCrossAttention, E3II_Layer

from DeepRL_3D import Transformer_lar, ModelArgs


class DeepRL_3D_model(nn.Module):
    r""" """

    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.args = args
        self.llama = Transformer_lar()######

        self.embedding = Embedding(args)
        self.vae = VariationalEncoder(args)

        if args.conditional:
            self.conditional = True
            self.num_cond_feature = args.num_cond_feature
        else:
            self.conditional = False
            self.num_cond_feature = 0

        if args.ssl:
            self.ssl_model = SSLModel(args)

        self.latent_mlp = nn.Linear(
            args.num_hidden_feature + args.num_latent_feature + self.num_cond_feature,
            args.num_hidden_feature,
        )

        self.next_type_ll = NextType(args, self.embedding.l_node_emb)
        self.next_type_lp = NextType(args, self.embedding.l_node_emb)
        self.next_dist_ll = NextDist(args, self.embedding.l_node_emb)
        self.next_dist_lp = NextDist(args, self.embedding.l_node_emb)

        self.loss_label_cla = nn.CrossEntropyLoss() #分类损失
        self.loss_label_reg = nn.MSELoss() #回归损失

        self.loss_fn = nn.KLDivLoss(reduction="none")
        if args.ssl:
            self.ssl_loss_fn = nn.CrossEntropyLoss(reduction="none")

        self.classifier = nn.Sequential(
            nn.Linear(128, 1024),  #compound_dim
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  #out_dim
        )

    def forward(
        self,
        data_dict,
        name_label_map,
        device
    ):
        r""" """
        # whole, traj, _ = data_dict.values()
        whole, traj, label_name = data_dict.values()
        # print("==========1111111===========",name_label_map)
        # exit()
        label_list = [max(0, name_label_map[key]) for key in label_name if key in name_label_map] #获取标签的值
        label = torch.tensor(label_list, dtype=torch.float32).to(device)  #将list转化为tensor
        # print("===========",label)
        # exit()

        if self.args.conditional:
            whole_cond = whole.pocket_prop
            traj_cond = traj.pocket_prop
        else:
            whole_cond = None
            traj_cond = None

        whole_ligand, whole_pocket = whole["ligand"].x, whole["pocket"].x


        # print("===========traj==============",traj["ligand"].x)
        # exit()

        # Embed(propagate) whole graph
        self.embedding(whole, cond=whole_cond)  #把维度embedding为128维，原来9维

        #print("======1111========",whole["ligand"])
        # ligand_indices_0 = (whole["ligand"].batch == 0)  # 第一个图的节点索引
        # ligand_indices_1 = (whole["ligand"].batch == 1)  # 第二个图的节点索引

        # ligand_graph_0 = whole["ligand"].h[ligand_indices_0]  # 第一个 ligand 图的节点特征
        # ligand_graph_1 = whole["ligand"].h[ligand_indices_1]  # 第二个 ligand 图的节点特征
        # print("======2222========",ligand_graph_0)
        # print("======3333========",ligand_graph_1)
        # print("======2222========",len(torch.bincount(whole["ligand"].batch))) #获取唯一的索引的数量，例如[0,0,0,0,1,1,1]-->[4,3]
        # unique_batches = torch.unique(whole["ligand"].batch)  #获取唯一的索引，例如[0,0,0,0,1,1,1]-->[0,1]
        # print("=========33333=====",unique_batches)
        # exit()

        # Sample latent vector and calculate vae loss
        h_cat_sub = torch.cat([whole["ligand"].h, whole["pocket"].h], 0)  # [L+P F]
        h_cat = h_cat_sub.mean(dim=0, keepdim=True)  # [1 F]

        latent, vae_loss = self.vae(whole, whole_ligand, whole_pocket)  #latent--[1,128]，进行预测的时候，返回latent
        latent_cat = h_cat #+ latent
        # latent_cat = latent
        # print("==================",latent_cat)
        # exit()

        # latent_pre = self.classifier(latent_cat)  ###用分类或者回归任务
        # print("=====latent_pre========",latent_pre, label)
        # print("=====llabel========",label)
        # exit()


        #loss_label_reg = self.loss_label_reg(latent_pre.squeeze(dim=0), label)  #加上标print("====1111=======",latent_pre)
        # loss_label_reg = self.loss_label_reg(latent_pre, label.unsqueeze(0))  #加上标print("====1111=======",latent_pre)
        # print("========1111=====",latent_pre)
        # print("========label====",label.unsqueeze(dim=0))
        # print("========2222=====",loss_label_reg)

        # print("====loss_label_reg=======",loss_label_reg)
        # print("====111=======",latent_pre.squeeze(dim=0))
        # print("====222=======",label)
        
        # print("=================",vae_loss)

        # Embed(propagate) unfinished graph
        self.embedding(traj, cond=traj_cond)  #把维度embedding为128维，原来9维
        # llama_logits_ligand = self.llama(traj["ligand"].h).mean(dim=2)
        # llama_logits_pocket = self.llama(traj["pocket"].h).mean(dim=2)
        # traj["ligand"].h = traj["ligand"].h + llama_logits_ligand
        # traj["pocket"].h = traj["pocket"].h + llama_logits_pocket



        # Concat latent vector with atom features
        if self.conditional:
            traj["pocket"].h = self.latent_mlp(
                torch.cat(
                    [
                        traj["pocket"].h,
                        traj_cond,
                        latent.repeat(traj["pocket"].h.shape[0], 1),
                    ],
                    -1,
                )
            )
        else:
            traj["pocket"].h = self.latent_mlp(
                torch.cat(
                    [traj["pocket"].h, latent.repeat(traj["pocket"].h.shape[0], 1)], -1
                )
            )

        # Predict p(Type|L) & p(Type|P)
        type_ll_pred = self.next_type_ll(traj, "ligand")
        type_lp_pred = self.next_type_lp(traj, "pocket")
        type_ll_loss = self.loss_fn(type_ll_pred, traj.type_output)
        type_lp_loss = self.loss_fn(type_lp_pred, traj.type_output)

        # Predict p(Position|L) & p(Position|P)
        dist_ll_pred = self.next_dist_ll(traj, traj.type_output, "ligand")
        dist_lp_pred = self.next_dist_lp(traj, traj.type_output, "pocket")
        ll_mask = (
            traj.mask[traj["ligand"].batch]
            .unsqueeze(-1)
            .repeat(1, self.args.dist_one_hot_param2[-1])
        )
        lp_mask = (
            traj.mask[traj["pocket"].batch]
            .unsqueeze(-1)
            .repeat(1, self.args.dist_one_hot_param2[-1])
        )
        dist_ll_loss = self.loss_fn(dist_ll_pred, traj.dist_ll_output) * ll_mask
        dist_lp_loss = self.loss_fn(dist_lp_pred, traj.dist_lp_output) * lp_mask

        dist_ll_loss = scatter_mean(dist_ll_loss, traj["ligand"].batch, 0)
        dist_lp_loss = scatter_mean(dist_lp_loss, traj["pocket"].batch, 0)

        vae_loss = self.args.vae_coeff * vae_loss.sum()  # KLDivLoss annealing
        type_ll_loss = type_ll_loss.mean(0).sum(0)  # Averaging in batch dimension
        type_lp_loss = type_lp_loss.mean(0).sum(0)  # Averaging in batch dimension
        dist_ll_loss = dist_ll_loss.sum() / traj.mask.sum()
        dist_lp_loss = dist_lp_loss.sum() / traj.mask.sum()

        type_loss = type_ll_loss + type_lp_loss
        dist_loss = dist_ll_loss + dist_lp_loss

        #print("=================",vae_loss)

        total_loss = vae_loss  + type_loss + dist_loss #+ loss_label_reg
        # total_loss = loss_label_reg

        # print("======loss_label_reg=========",loss_label_reg)

        if self.args.ssl:
            cond_pred = self.ssl_model(whole)
            ssl_loss = self.ssl_loss_fn(cond_pred, whole_cond.argmax(-1)).mean()
            total_loss += ssl_loss
            return total_loss, vae_loss, type_loss, dist_loss, ssl_loss #, latent_pre, loss_label_reg

        return total_loss, vae_loss, type_loss, dist_loss, None #, latent_pre, loss_label_reg


class Embedding(nn.Module):
    r"""
    Node embedding and propagation
    """

    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.args = args
        self.llama = Transformer_lar()

        self.l_node_emb = nn.Sequential(
            nn.Linear(args.num_ligand_atom_feature, args.num_hidden_feature),
        )
        self.p_node_emb = nn.Sequential(
            nn.Linear(args.num_pocket_atom_feature, args.num_hidden_feature),
        )
        self.emb_dict = {"ligand": self.l_node_emb, "pocket": self.p_node_emb}

        self.layers = nn.ModuleList([E3II_Layer(args) for _ in range(args.num_layers)])

    def forward(self, data, cond=None):
        data["ligand"].h = self.l_node_emb(data["ligand"].x)
        data["pocket"].h = self.p_node_emb(data["pocket"].x)

        # llama_logits_ligand = self.llama(data["ligand"].x)
        # llama_logits_pocket = self.llama(data["pocket"].x)

        for lay in self.layers:
            lay(data, cond=cond)

        # print("===================",data["ligand"].h.shape)
        # exit()

        llama_logits_ligand = self.llama(data["ligand"].h).mean(dim=2)
        llama_logits_pocket = self.llama(data["pocket"].h).mean(dim=2)
        data["ligand"].h = data["ligand"].h + llama_logits_ligand.detach()  #data["ligand"].h + llama_logits_ligand-->存在就地操作,应修改为：data["ligand"].h + llama_logits_ligand.detach()
        data["pocket"].h = data["pocket"].h + llama_logits_pocket.detach()

        # print("===============",data["ligand"].h.shape)
        # exit()

        # data["ligand"].h = data["ligand"].h #+ llama_logits_ligand
        # data["pocket"].h = data["pocket"].h #+ llama_logits_pocket

        return data


class NextType(nn.Module):
    r"""
    Predict p(Type)
    """

    def __init__(self, args, embedding=None):
        super().__init__()

        self.args = args
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Linear(
                args.num_ligand_atom_feature, args.num_hidden_feature, bias=False
            )

        self.act = nn.SiLU()
        self.last_act = None

        layers = []
        n_dims = list(
            np.linspace(args.num_hidden_feature, 1, args.num_dense_layers + 1).astype(
                int
            )
        )
        for n_in, n_out in zip(n_dims[:-1], n_dims[1:]):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(self.act)
        layers = layers[:-1]
        if self.last_act is not None:
            layers.append(self.last_act)

        self.dense = nn.Sequential(*layers)

        self.atom_type = torch.eye(args.num_ligand_atom_feature)
        self.atom_type = nn.Parameter(self.atom_type)
        self.atom_type.requires_grad = False

    def forward(self, data, key="ligand"):
        r"""
        Args:
            data (torch_geometric.data.HeteroDataBatch)

        Returns:
            next_type (torch.Tensor): [num_type]
        """

        embed_type = self.embedding(self.atom_type)  # [num_type, num_hidden]
        embed_type = embed_type.unsqueeze(0)  # [1, num_type, num_hidden]
        repr_type = data[key].h.unsqueeze(1)  # [N, 1, num_hidden]

        mul_type = embed_type * repr_type  # [N, num_type, num_hidden]
        dense_type = self.dense(mul_type).squeeze(-1)  # [N, num_type]

        batch = data[key].batch
        next_type = F.log_softmax(dense_type, dim=-1)  # [N, num_type]
        next_type_agg = scatter_add(next_type, batch, dim=0)  # [B, num_type]
        next_type_agg = next_type_agg - torch.logsumexp(
            next_type_agg, dim=-1, keepdim=True
        )

        return next_type_agg


class NextDist(nn.Module):
    r"""
    Predict p(Distance)
    """

    def __init__(self, args, embedding=None):
        super().__init__()

        self.args = args
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Linear(
                args.num_ligand_atom_feature, args.num_hidden_feature, bias=False
            )

        self.act = nn.SiLU()
        self.last_act = None

        layers = []
        n_dims = list(
            np.linspace(
                args.num_hidden_feature,
                args.dist_one_hot_param2[-1],
                args.num_dense_layers + 1,
            ).astype(int)
        )
        for n_in, n_out in zip(n_dims[:-1], n_dims[1:]):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(self.act)
        layers = layers[:-1]
        if self.last_act is not None:
            layers.append(self.last_act)

        self.dense = nn.Sequential(*layers)

        self.use_attention = False  # TODO
        if self.use_attention:
            # Constrained cross attention from E3Bind
            self.attn = ConstrainedCrossAttention(args)

    def forward(self, data, next_type, key):  # "ligand" or "pocket"
        batch = data[key].batch
        type_embed = self.embedding(next_type)[batch]  # [N, num_hidden]
        dist_embed = data[key].h * type_embed
        if self.use_attention:
            dist_embed, attn = self.attn(dist_embed, data[key].pos, batch)
        next_dist = F.log_softmax(self.dense(dist_embed), dim=-1)
        return next_dist

class ContrastiveLoss(nn.Module):
    def forward(self, output1, output2):
        distance = torch.norm(output1 - output2, p=2, dim=1)
        # 这里假设你有一个阈值来定义相似性
        margin = 1.0
        return torch.mean(torch.relu(margin - distance))

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()

        # for param in self.fc1.parameters():
        #     param.requires_grad = False
        # for param in self.fc_mu.parameters():
        #     param.requires_grad = False
        # for param in self.fc_logvar.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
# 定义解码器
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # for param in self.fc1.parameters():
        #     param.requires_grad = False
        # for param in self.fc2.parameters():
        #     param.requires_grad = False

    def forward(self, z):
        h = self.relu(self.fc1(z))
        x_hat = self.sigmoid(self.fc2(h))
        return x_hat


class VariationalEncoder(nn.Module):
    r""" """

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.llama = Transformer_lar()######

        self.mean = nn.Linear(args.num_hidden_feature, args.num_latent_feature)
        self.logvar = nn.Linear(args.num_hidden_feature, args.num_latent_feature)

        self.encoder = Encoder(args.num_hidden_feature, args.num_latent_feature, args.num_latent_feature)
        self.decoder = Decoder(args.num_latent_feature, args.num_latent_feature, args.num_hidden_feature)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, device=std.device)
        return eps * std + mean

    def vae_loss(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
        # cos_sim = F.cosine_similarity(x, x_hat) #[-1,1]
        # cos_sim_loss = 1 -cos_sim #通过1-，[0,2]可以衡量损失的大小，相似度越高损失越小，相似度越低损失越大

    def vae_loss_1(self, x, x_hat, mean, logvar):
        recon_loss = F.binary_cross_entropy(x_hat, torch.sigmoid(x)) #重构损失
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1) #KL散度损失
        vae_loss = recon_loss + kl_div
        return vae_loss


    def vae_loss_sub(self, mean, logvar):
        #return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
        cos_sim = F.cosine_similarity(mean, logvar) #[-1,1]
        cos_sim_loss = 1 -cos_sim #通过1-，[0,2]可以衡量损失的大小，相似度越高损失越小，相似度越低损失越大
        return cos_sim_loss

    def vae_loss_sub_sub(self, output1, output2):
        distance = torch.norm(output1 - output2, p=2, dim=1)
        # print("==========output1====",output1)
        # print("==========output2====",output2)
        # print("==========distance====",distance)
        # 这里假设有一个阈值来定义相似性
        margin = 2.0
        return torch.mean(torch.relu(margin - distance))

    def forward(
        self,
        data,
        whole_ligand, 
        whole_pocket,
    ):
        
        # llama_logits_ligand = self.llama(data["ligand"].h).mean(dim=2)
        # llama_logits_pocket = self.llama(data["pocket"].h).mean(dim=2)
        # print("========h_cat_sub==========",llama_logits_ligand.shape)
        # print("========h_cat_sub==========",llama_logits_pocket.shape)
        # exit()

        #h_cat_sub = torch.cat([llama_logits_ligand, llama_logits_pocket], 0)
        h_cat = torch.cat([data["ligand"].h, data["pocket"].h], 0)  # [L+P F]
        h_cat = h_cat #+ h_cat_sub

        readout = h_cat.mean(dim=0, keepdim=True)  # [1 F]
        mean, logvar = self.encoder(readout)

        # mean = self.mean(readout)
        # logvar = self.logvar(readout)

        latent = self.reparameterize(mean, logvar)  # [1 F']
        x_hat = self.decoder(latent)

        #vae_loss = self.vae_loss(mean, logvar)
        vae_loss = self.vae_loss_1(readout, x_hat, mean, logvar) 


        return latent, vae_loss


class SSLModel(nn.Module):
    def __init__(self, args, embedding=None):
        super().__init__()

        self.args = args

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Linear(
                args.num_pocket_atom_feature, args.num_hidden_feature, bias=False
            )

        self.num_layers = args.num_dense_layers
        self.layers = nn.ModuleList([EGCL(args) for _ in range(self.num_layers)])
        self.fc_layer = nn.Linear(args.num_hidden_feature, args.num_cond_feature)

    def forward(self, data):
        edge_index_ = data["p2p"].edge_index.clone()
        h_ = data["pocket"].x.clone()
        x_ = data["pocket"].pos.clone()

        for layer in self.layers:
            h_, x_ = layer(h_, x_, edge_index_)

        y = self.fc_layer(h_)
        return y