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

txt_file="davis_mutilable/davis_kinase.csv"

class Datasets(nn.Module):
    def __init__(self):
        drug_data = [
            np.load(f"{txt_file}_x.npy", allow_pickle=True),

            np.load(f"{txt_file}_a.npy", allow_pickle=True)
        ]
        self.drug_smiles=np.load(f"{txt_file}_token.npy", allow_pickle=True)
        self.drug_atom=drug_data[0]
        self.drug_adj=drug_data[1]
        self.drug_graph=[]
        for i in range(len(self.drug_smiles)):
               graph=dgl.from_scipy(sp.coo_matrix(self.drug_adj[i]))
               graph.ndata['feat']=torch.tensor(self.drug_atom[i])
               self.drug_graph.append(graph)

        self.protein=np.load(f"{txt_file}_protein.npy", allow_pickle=True)
        self.labels=np.load(f"{txt_file}_y.npy", allow_pickle=True)
        self.index=np.load(f"{txt_file}_index.npy", allow_pickle=True)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):

        return self.drug_graph[item],self.drug_smiles[item],self.protein[item],self.labels[item],self.index[item]


class RGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_rel):
        super().__init__()
        self.RGCNlayers=nn.ModuleList()

        num_layers=4
        for layer in range(num_layers-1):
            if layer==0:
                self.RGCNlayers.append(
                     RelGraphConv(input_dim,hidden_dim,num_rel)
                )
            else:
                self.RGCNlayers.append(
                    RelGraphConv(hidden_dim,hidden_dim,num_rel)
                )

        self.mtp=nn.Sequential(
            nn.Linear( 2000 ,  2048),

            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(2048),

            nn.Linear(2048,1024),
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
            nn.Linear(128,1)

        )

    def forward(self, g, feat, etype, index, device):
        for i, layer in enumerate(self.RGCNlayers):
            feat = layer(g, feat, etype)

        index=index.permute(1,0)


        drug=feat[index[0]]

        protein=feat[index[1]]

        drug_protein=torch.cat((drug,protein),-1)


        dti_predict = self.mtp(drug_protein.to(device))
        return dti_predict,feat

def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i,data in enumerate(Y):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair is not 0:
        return summ / pair
    else:
        return 0


def evaluate_RGCN(graph,feat,etype,index,label, device, model):
    model.eval()

    graph = graph.to(device)
    feature = torch.tensor(np.array(feat),dtype=torch.float32).to(device)
    etype = torch.as_tensor(etype).to(device)
    index = torch.as_tensor(index,dtype=torch.long).to(device)
    logits,_ = model(graph, feature, etype, index, device)
    logits = logits.cpu().detach().numpy()

    total_loss = mean_squared_error(logits, label)
    c_index=get_cindex(label,logits)

    return total_loss,float(c_index)

def get_rgcn_feature(graph,feat,etype,index,device,model):
    model.eval()
    graph = graph.to(device)
    feature = torch.tensor(feat, dtype=torch.float32).to(device)
    etype = torch.as_tensor(etype).to(device)
    index = torch.as_tensor(index,dtype=torch.long).to(device)
    _,rgcn_feature= model(graph, feature, etype, index, device)

    return rgcn_feature


f=open("log_davis_RGCN.txt", 'w')


dgl0 = np.load(f"{txt_file}_dgl0.npy",allow_pickle=True)
dgl1 = np.load(f"{txt_file}_dgl1.npy", allow_pickle=True)
etype = np.load(f"{txt_file}_edg_type.npy", allow_pickle=True)
index=np.load(f"{txt_file}_index.npy", allow_pickle=True)
adj=np.load(f"{txt_file}_adj.npy", allow_pickle=True)

labels=np.load(f"{txt_file}_y.npy", allow_pickle=True)
all_data=Datasets()
all_num=all_data.__len__()
data_induce=np.arange(0,all_num)
kf=KFold(n_splits=5,shuffle=True,random_state=100)
data_loaders=dict()
j=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for train_index,val_index in kf.split(data_induce):

    data_loaders_RGCN = GraphDataLoader(all_data, pin_memory=torch.cuda.is_available())

    etype_mask = etype[train_index]
    g_val=dgl.graph((dgl0,dgl1))
    g=dgl.graph((dgl0[train_index],dgl1[train_index]))

    np.random.seed(100)
    num=len(adj)
    feat=[list() for i in range(num)]
    all_pre=np.random.randn(all_num,2100)



    tra_label = labels[train_index]
    tra_index = index[train_index]
    val_label = labels[val_index]
    val_indice = index[val_index]


    for i,data in enumerate(index):
        if len(feat[data[0]])==0:
            feat[data[0]]=all_pre[i]
        if len(feat[data[1]])==0:
            feat[data[1]]=all_pre[i]
    epoch_rgcn = 6000
    model_2 = RGCN(2100, 1000, 6).to(device)

    loss_model_2 = nn.MSELoss()
    optim_2 = torch.optim.Adam(model_2.parameters(), weight_decay=1e-4, lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optim_2, step_size=1800, gamma=0.1)
    num_rgcn=0
    min_loss=1000

    for i in range(epoch_rgcn):
        model_2.train()

        graph=g.to(device)
        feature=torch.tensor(np.array(feat),dtype=torch.float32).to(device)
        etype_g=torch.as_tensor(etype_mask).to(device)
        label=torch.tensor(np.array(tra_label)).to(device)
        index_g=torch.as_tensor(np.array(tra_index),dtype=torch.long).to(device)
        logits,_=model_2(graph,feature,etype_g,index_g,device)
        loss = loss_model_2(logits, label)
        optim_2.zero_grad()
        loss.backward()
        optim_2.step()

        if num_rgcn>2000:
            if num_rgcn<=5000 and num_rgcn%100==0:

                f.write("RGCN LOSS: {:.4f}\n".format(loss.data.item()))
                val_acc, c_index = evaluate_RGCN(g_val, feat, etype, val_indice, val_label, device, model_2)
                f.write("Validation MSE AND CI. {:.4f} {:.4f}\n".format(val_acc, c_index))
                print(
                    " Validation MSE. {:.4f} .  ".format(
                        val_acc
                    ))
                print(c_index)

                if min_loss>val_acc:
                    min_loss=val_acc
                    torch.save(model_2,"./weight_davis_rgcn/"+str(j)+"_bestmodel.pkl")
            elif num_rgcn>5000 and num_rgcn %5==0:
                val_acc, c_index = evaluate_RGCN(g_val, feat, etype, val_indice, val_label, device, model_2)
                f.write("last 1000, Validation MSE AND CI. {:.4f} {:.4f}\n".format(val_acc, c_index))
                print(
                    " Validation MSE. {:.4f} .  ".format(
                        val_acc
                    ))
                print(c_index)

                if min_loss > val_acc:
                    min_loss = val_acc
                    torch.save(model_2, "./weight_davis_rgcn/" + str(j) + "_bestmodel.pkl")

            
        print(loss.data.item())


    val_acc,c_index=evaluate_RGCN(g,feat,etype_mask,val_indice,val_label,device,model_2)
    f.write( "each fold RGCN Validation mse and ci. {:.4f} . \n {:.4f}\n".format(
            val_acc,float(c_index)
        ))
    print(
        " Validation mse. {:.4f} .  ".format(
            val_acc
        ))
    print(c_index)
    j = j + 1
f.close()















