import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.conv import RelGraphConv
from dgl.nn.pytorch.glob import SumPooling
import numpy as np
from sklearn.model_selection import KFold
import dgl
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
from dgl.dataloading import GraphDataLoader
import random
import json
import os
from sklearn.metrics import r2_score
import esm
from transformers import AutoModelForMaskedLM



class CoAttentionLayer(nn.Module):
    def __init__(self, hidden_dim=480):
        super(CoAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
    def forward(self, text_features, image_features):
        print(text_features.size())
        print(image_features.size())
        batch_size, token_num, _ = text_features.size()
        _, img_feat_num, _ = image_features.size()
        device = text_features.device

        attn_scores = torch.bmm(text_features, image_features.permute(0, 2, 1))

        top_k = int(image_features.size(1) * 0.3)

        top_values, top_indices = torch.topk(attn_scores, top_k, dim=-1)

        batch_indices = torch.arange(top_indices.size(0)).view(-1, 1, 1).expand_as(top_indices).to(device)
        token_indices = torch.arange(top_indices.size(1)).view(1, -1, 1).expand_as(top_indices).to(device)
        top_indices_qk = torch.stack([batch_indices, token_indices, top_indices], dim=-1)
        print(top_indices_qk.size())

        attn_weights_text = F.softmax(attn_scores, dim=-1)
        attn_weights_image = F.softmax(attn_scores.permute(0, 2, 1), dim=-1)
        context_text = torch.bmm(attn_weights_text, image_features)
        context_image = torch.bmm(attn_weights_image, text_features)

        return context_text, context_image, attn_weights_text

txt_file = "davis_mutilable/davis_kinase.csv"
adj = np.load(f"{txt_file}_adj.npy", allow_pickle=True)
All_num = len(adj)


class Datasets(nn.Module):
    def __init__(self):
        drug_data = [
            np.load(f"{txt_file}_x.npy", allow_pickle=True),

            np.load(f"{txt_file}_a.npy", allow_pickle=True)
        ]
        self.drug_smiles = np.load(f"{txt_file}_token.npy", allow_pickle=True)
        self.drug_atom = drug_data[0]
        self.drug_adj = drug_data[1]

        self.protein = np.load(f"{txt_file}_protein.npy", allow_pickle=True)
        self.labels = np.load(f"{txt_file}_y.npy", allow_pickle=True)
        self.index = np.load(f"{txt_file}_index.npy", allow_pickle=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.drug_smiles[item], self.protein[item]


class Datasets2(nn.Module):
    def __init__(self):
        drug_data = [
            np.load(f"{txt_file}_x.npy", allow_pickle=True),

            np.load(f"{txt_file}_a.npy", allow_pickle=True)
        ]
        file_name1 = "davis_mutilable/davis_kinase.csv_drug_x.json"
        with open(file_name1) as file_obj:
            drug_x = json.load(file_obj)
        file_name2 = "davis_mutilable/davis_kinase.csv_drug_a.json"
        with open(file_name2) as file_obj:
            drug_a = json.load(file_obj)

        self.drug_smiles = np.load(f"{txt_file}_drug_embedding.npy", allow_pickle=True)
        self.drug_atom = drug_x
        self.drug_adj = drug_a
        self.drug_graph = []
        for i in range(len(self.drug_adj)):
            graph = dgl.from_scipy(sp.coo_matrix(self.drug_adj[i]))
            graph.ndata['feat'] = torch.tensor(self.drug_atom[i])
            self.drug_graph.append(graph)
        self.index = np.load(f"{txt_file}_index.npy", allow_pickle=True)
        file_name = "davis_kinase.csv_protein_a_x.json"
        with open(file_name) as file_obj:
            protein_x_a = json.load(file_obj)
        self.protein_x = []
        self.protein_a = []

        protein_num = len(protein_x_a)
        self.drug_num = All_num - protein_num

        self.protein_graph = []
        for i in range(len(protein_x_a)):
            protein_adj = np.array(protein_x_a[i][1])
            protein_adj = protein_adj.transpose(1, 0)
            src_ids = torch.tensor(protein_adj[0])
            dst_ids = torch.tensor(protein_adj[1])
            graph = dgl.graph((src_ids, dst_ids))
            graph.ndata['feat'] = torch.tensor(protein_x_a[i][0], dtype=torch.float32)
            self.protein_graph.append(graph)

        self.protein_ami = np.load(f"{txt_file}_protein_embedding.npy", allow_pickle=True)
        self.labels = np.load(f"{txt_file}_y.npy", allow_pickle=True)
        self.drug_rgcn = np.load(f"{txt_file}_drug_rgcn_c2.npy", allow_pickle=True)
        self.protein_rgcn = np.load(f"{txt_file}_protein_rgcn_c2.npy", allow_pickle=True)
        self.num = len(self.labels)

        self.random_feat = torch.tensor(np.random.randn(self.num, 500), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):

        return self.drug_graph[self.index[item][0]], self.drug_smiles[self.index[item][0]], self.protein_graph[
            self.index[item][1] - self.drug_num], self.protein_ami[self.index[item][1] - self.drug_num], self.labels[
                   item], self.drug_rgcn[item], \
               self.protein_rgcn[item], self.random_feat[item]


class RGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_rel):
        super().__init__()
        self.RGCNlayers = nn.ModuleList()

        num_layers = 4
        for layer in range(num_layers - 1):
            if layer == 0:
                self.RGCNlayers.append(
                    RelGraphConv(input_dim, hidden_dim, num_rel)
                )
            else:
                self.RGCNlayers.append(
                    RelGraphConv(hidden_dim, hidden_dim, num_rel)
                )

        self.mtp = nn.Sequential(
            nn.Linear(2000, 2048),

            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)

        )

    def forward(self, g, feat, etype, index, device):
        # f_unprocess = feat.tolist()
        for i, layer in enumerate(self.RGCNlayers):
            feat = layer(g, feat, etype)
            # feat=self.batch_norms[i](feat)
        # print(feat.shape)  #443 X 300
        # print('--------------------------------')
        index = index.permute(1, 0)

        drug = feat[index[0]]

        protein = feat[index[1]]

        drug_protein = torch.cat((drug, protein), -1)

        dti_predict = self.mtp(drug_protein.to(device))
        # dti_predict = self.mtp(feat)
        return dti_predict, feat


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


# 5层 过多 一般2-3 ，可学习参数E启用应该可以累加层数
class GIN_GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, input_dim_2, hidden_dim_2):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.ginlayers_2 = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.batch_norms_2 = nn.ModuleList()

        num_layers_1 = 3
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers_1 - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        num_layers_2 = 3
        for layer in range(num_layers_2 - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim_2, hidden_dim_2, hidden_dim_2)
            else:
                mlp = MLP(hidden_dim_2, hidden_dim_2, hidden_dim_2)
            self.ginlayers_2.append(
                GINConv(mlp, learn_eps=False)
            )
            self.batch_norms_2.append(nn.BatchNorm1d(hidden_dim_2))

        self.pool = (
            SumPooling()
        )
        self.cross_model1 = CoAttentionLayer(hidden_dim=480)
        self.cross_model2 = CoAttentionLayer(hidden_dim=480)
        self.cross_model3 = CoAttentionLayer(hidden_dim=480)

        self.layern_d_g_s = nn.LayerNorm(256)
        self.layern_p_g_s = nn.LayerNorm(480)
        self.layern_pd_s_s = nn.LayerNorm(960)

        self.rgcn_2 = nn.Linear(2000, 512)
        self.layern_rgch=nn.LayerNorm(512)

        self.layern = nn.LayerNorm(512)
        self.mtp1 = nn.Sequential(
            # nn.BatchNorm1d(1112),
            nn.Linear(1696, 2048),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.LayerNorm(2048),
            nn.Linear(2048, 512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 32),
            nn.Dropout(0.1),
            nn.ReLU(),
            # nn.LayerNorm(32),
            nn.Linear(32, 1)

        )
        self.drop = nn.Dropout(0.1)
        # -----------------------------------------

    def forward(self, g, h, g_2, h_2, drug, protein, drug_rgcn, protein_rgcn, random_feat, batch):
        # list of hidden representation at each layer (including the input layer)



        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        hidden_rep_2 = [h_2]
        for i, layer in enumerate(self.ginlayers_2):
            h_2 = layer(g_2, h_2)
            h_2 = self.batch_norms_2[i](h_2)
            h_2 = F.relu(h_2)
            hidden_rep_2.append(h_2)
        #
        # 整个图的表示，因为全部节点的特征作sum,使用药物的话就是整个药物的特征提取
        drug_rep = 0
        for i, h in enumerate(hidden_rep):
            if i == 0: continue
            pooled_h = self.pool(g, h)
            drug_rep += self.drop(pooled_h)

        protein_rep = 0
        for i, h in enumerate(hidden_rep_2):
            if i == 0: continue
            pooled_h = self.pool(g_2, h)
            protein_rep += self.drop(pooled_h)
        g.ndata['feat']=hidden_rep[-1]
        g_2.ndata['feat']=hidden_rep_2[-1]
        g_unbatch=dgl.unbatch(g)
        g_2unbatch=dgl.unbatch((g_2))
        drug_node=[]
        for i in g_unbatch:
            t=i.ndata.pop("feat")
            if t.size(0)<200:
                t=F.pad(t, (0, 0, 0, 200-t.size(0)))
            else:
                t=t[:200,:]
            drug_node.append(t)
        drug_node = torch.stack(drug_node, dim=0)
        # print(drug_node.size())   batchsize,200,256
        protein_node=[]
        for i in g_2unbatch:
            t = i.ndata.pop("feat")
            if t.size(0) < 200:
                t = F.pad(t, (0, 0, 0, 200 - t.size(0)))
            else:
                t = t[:200, :]
            protein_node.append(t)

        protein_node=torch.stack(protein_node,dim=0)

        # print(protein_node.size())  batchsize,200,512
        # X_split = torch.split(X, num_nodes_per_graph)

        #hidden_rep[-1]  batch*node*feature  然后与序列做co attention


        graph_seq_drug,_,_=self.cross_model1(drug_node, drug[:,:,:256])
        graph_seq_protein, _, _ = self.cross_model2(protein_node[:,:,:480], protein[:,:200,:])
        graph_seq_drug = torch.mean(graph_seq_drug, dim=1)
        graph_seq_drug = self.layern_d_g_s(graph_seq_drug)

        graph_seq_protein = torch.mean(graph_seq_protein, dim=1)
        graph_seq_protein = self.layern_p_g_s(graph_seq_protein)
        graph_seq_fusion=torch.cat((graph_seq_drug,graph_seq_protein),-1)



        # rgcn_h = torch.cat((drug_rgcn, protein_rgcn), -1)
        # rgcn_h = self.rgcn_2(rgcn_h)
        # rgcn_h=self.layern_rgch(rgcn_h)

        context_drugs,context_protein,_=self.cross_model3(drug,protein[:,:200,:])
        pre_drug=torch.mean(context_drugs,dim=1)
        pre_protein=torch.mean(context_protein,dim=1)
        pre_seq=torch.cat((pre_drug,pre_protein),-1)
        pre_seq = self.layern_pd_s_s(pre_seq)

        final_feature = torch.cat((graph_seq_fusion, pre_seq), -1)
        # final_feature =torch.cat((final_feature,rgcn_h),-1)
        dti_pre = self.mtp1(final_feature)
        # dti_pre = torch.sigmoid(dti_pre) * 17.2

        return dti_pre



def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i, data in enumerate(Y):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair is not 0:
        return summ / pair
    else:
        return 0


def evaluate(dataloader, device, model):
    model.eval()
    total = 0
    num = 0
    c_index = 0
    r_2 = 0
    for train_drug_graph, train_drug_smiles, train_protein_graph, train_protein_ami, train_label, train_drug_rgcn, train_protein_rgcn, train_random_feat in dataloader:
        train_drug_graph = train_drug_graph.to(device)
        train_protein_graph = train_protein_graph.to(device)
        train_drug_rgcn = train_drug_rgcn.to(device)
        train_protein_rgcn = train_protein_rgcn.to(device)
        train_drug_smiles = train_drug_smiles.to(device)
        train_protein_ami = train_protein_ami.to(device)
        train_label = train_label.to(device)
        train_drug_atom = train_drug_graph.ndata.pop("feat")
        train_protein_red = train_protein_graph.ndata.pop("feat")
        train_random_feat = train_random_feat.to(device)

        logits = model_1(train_drug_graph, train_drug_atom, train_protein_graph, train_protein_red, train_drug_smiles,
                         train_protein_ami, train_drug_rgcn, train_protein_rgcn, train_random_feat, len(train_label))
        logits = logits.cpu().detach().numpy()
        train_label = train_label.cpu()
        total += mean_squared_error(logits, train_label)
        c_index += float(get_cindex(train_label, logits))
        r_2 += r2_score(train_label, logits)
        num += 1

    return total / num, c_index / num, r_2 / num


def evaluate_RGCN(graph, feat, etype, index, label, device, model):
    model.eval()

    graph = graph.to(device)
    feature = torch.tensor(feat, dtype=torch.float32).to(device)
    etype = torch.as_tensor(etype).to(device)
    index = torch.as_tensor(index, dtype=torch.long).to(device)
    logits, _ = model(graph, feature, etype, index, device)  # 返回各个节点特征   443维
    logits = logits.cpu().detach().numpy()

    total_loss = mean_squared_error(logits, label)
    c_index = get_cindex(label, logits)

    return total_loss, float(c_index)


def get_rgcn_feature(graph, feat, etype, index, device, model):
    model.eval()
    graph = graph.to(device)
    feature = torch.tensor(feat, dtype=torch.float32).to(device)
    etype = torch.as_tensor(etype).to(device)
    index = torch.as_tensor(index, dtype=torch.long).to(device)
    _, rgcn_feature = model(graph, feature, etype, index, device)

    return rgcn_feature.cpu().detach().numpy()


# f = open("log_davis_GIN_c_2.txt", 'w')

dgl0 = np.load(f"{txt_file}_dgl0.npy", allow_pickle=True)
dgl1 = np.load(f"{txt_file}_dgl1.npy", allow_pickle=True)
etype = np.load(f"{txt_file}_edg_type.npy", allow_pickle=True)
index = np.load(f"{txt_file}_index.npy", allow_pickle=True)

labels = np.load(f"{txt_file}_y.npy", allow_pickle=True)
all_data = Datasets()
all_num = all_data.__len__()
data_induce = np.arange(0, all_num)
kf = KFold(n_splits=5, shuffle=True, random_state=100)
data_loaders = dict()
j = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for train_index, val_index in kf.split(data_induce):
    print("----第", j, "折训练----")
    data_loaders_RGCN = GraphDataLoader(all_data, pin_memory=torch.cuda.is_available())
    etype_mask = etype[train_index]
    g = dgl.graph((dgl0[train_index], dgl1[train_index]))
    g_val = dgl.graph((dgl0, dgl1))
    drug_pre = []
    protein_pre = []
    t1 = np.zeros((1, 2000))
    t2 = np.zeros((1, 100))

    for train_drug_smiles, train_protein in data_loaders_RGCN:
        drug_pre.append(np.concatenate((train_drug_smiles, t1), -1).squeeze())
        protein_pre.append(np.concatenate((t2, train_protein), -1).squeeze())
    np.random.seed(100)
    num = len(adj)
    feat = [list() for i in range(num)]
    all_pre = np.random.randn(all_num, 2100)
    tra_label = labels[train_index]
    tra_index = index[train_index]
    val_label = labels[val_index]
    val_indice = index[val_index]

    for i, data in enumerate(index):
        if len(feat[data[0]]) == 0:
            feat[data[0]] = all_pre[i]
        if len(feat[data[1]]) == 0:
            feat[data[1]] = all_pre[i]
    print('-------------get graph feature-------------')
    best_model = torch.load("./weight_davis_rgcn/" + str(j) + "_bestmodel.pkl")
    rgcn_feature = get_rgcn_feature(g_val, feat, etype, tra_index, device, best_model)
    drug_rgcn_feature = []
    protein_rgcn_feature = []
    for i, data in enumerate(index):
        d = rgcn_feature[data[0]]
        p = rgcn_feature[data[1]]
        drug_rgcn_feature.append(d)
        protein_rgcn_feature.append(p)
    print(len(drug_rgcn_feature))
    print(len(protein_rgcn_feature))
    d_r = np.array(drug_rgcn_feature, dtype=np.float32)
    p_r = np.array(protein_rgcn_feature, dtype=np.float32)
    np.save(f"{txt_file}_drug_rgcn_c2.npy", d_r)
    np.save(f"{txt_file}_protein_rgcn_c2.npy", p_r)
    torch.cuda.empty_cache()
    all_gin_data = Datasets2()
    train_subset = torch.utils.data.Subset(all_gin_data, train_index)
    val_subset = torch.utils.data.Subset(all_gin_data, val_index)
    data_loaders['train'] = GraphDataLoader(train_subset, batch_size=500, pin_memory=torch.cuda.is_available(),
                                            shuffle=True)
    data_loaders['val'] = GraphDataLoader(val_subset, batch_size=500, pin_memory=torch.cuda.is_available(),
                                          shuffle=True)
    print('----------------GIN training----------------')
    model_1 = GIN_GIN(34, 256, 54, 512).to(device)
    min_each_loss = 10000
    loss_model = nn.MSELoss()
    optim = torch.optim.Adam(model_1.parameters(),weight_decay=1e-4, lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=250, gamma=0.1)
    num_print = 0
    epoch_getfeature = 650
    for i in range(epoch_getfeature):
        model_1.train()
        total_loss = 0
        for batch, (
                train_drug_graph, train_drug_smiles, train_protein_graph, train_protein_ami, train_label,
                train_drug_rgcn,
                train_protein_rgcn, train_random_feat) in enumerate(data_loaders['train']):
            # train_drug_atom,train_drug_graph,train_drug_smiles,train_protein,train_label=train
            train_drug_graph = train_drug_graph.to(device)
            train_protein_graph = train_protein_graph.to(device)
            train_drug_rgcn = train_drug_rgcn.to(device)
            train_protein_rgcn = train_protein_rgcn.to(device)
            train_drug_smiles = train_drug_smiles.to(device)
            train_protein_ami = train_protein_ami.to(device)
            train_label = train_label.to(device)
            train_drug_atom = train_drug_graph.ndata.pop("feat")
            train_protein_red = train_protein_graph.ndata.pop("feat")
            train_random_feat = train_random_feat.to(device)
            logits = model_1(train_drug_graph, train_drug_atom, train_protein_graph, train_protein_red,
                             train_drug_smiles, train_protein_ami, train_drug_rgcn, train_protein_rgcn,
                             train_random_feat, len(train_label))
            loss = loss_model(logits, train_label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            print(loss.data.item())
        scheduler.step()

        if 40<=num_print <= 400 and num_print % 20 == 0:
            train_acc, ci_trian, train_r_2 = evaluate(data_loaders['train'], device, model_1)
            valid_acc, ci_valid, valid_r_2 = evaluate(data_loaders['val'], device, model_1)
            # f.write(
            #     "Epoch {:05d} | Loss {:.4f}  Train Mse. {:.4f}  Train CI. {:.4f} Train r_2. {:.4f} | Validation Mse {:.4f}   Validation CI. {:.4f} Validation r_2. {:.4f} \n".format(
            #         i, total_loss / len(train_label), train_acc, ci_trian, train_r_2, valid_acc, ci_valid, valid_r_2))
            print(
                "Epoch {:05d} | Loss {:.4f} | Train Mse. {:.4f} | Train CI. {:.4f} | Validation Mse. {:.4f} |  Validation CI. {:.4f} | Validation r_2. {:.4f}".format(
                    i, total_loss / len(train_label), train_acc, ci_trian, valid_acc, ci_valid, valid_r_2
                ))
        if 500<num_print and num_print % 5 == 0:
            train_acc, ci_trian, train_r_2 = evaluate(data_loaders['train'], device, model_1)
            valid_acc, ci_valid, valid_r_2 = evaluate(data_loaders['val'], device, model_1)
            # f.write(
            #     "Epoch {:05d} | Loss {:.4f}  Train Mse. {:.4f}  Train CI. {:.4f} Train r_2. {:.4f} | Validation Mse {:.4f}   Validation CI. {:.4f} Validation r_2. {:.4f} \n".format(
            #         i, total_loss / len(train_label), train_acc, ci_trian, train_r_2, valid_acc, ci_valid, valid_r_2))
            print(
                "Epoch {:05d} | Loss {:.4f} | Train Mse. {:.4f} | Train CI. {:.4f} | Validation Mse. {:.4f} |  Validation CI. {:.4f} | Validation r_2. {:.4f}".format(
                    i, total_loss / len(train_label), train_acc, ci_trian, valid_acc, ci_valid, valid_r_2
                ))

        num_print += 1

    j = j + 1
# f.close()















