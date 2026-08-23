import json

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

num_atom_feat = 34

def protein_seq_to_idx_mask(seq, vocab_dict, max_len=2000):
    if len(seq) > max_len:
        seq = seq[:max_len]
    
    token = np.zeros(max_len, dtype=np.float32)
    mask = np.ones(max_len, dtype=np.float32)
    for i, char in enumerate(seq):
        idx = vocab_dict.get(char, 0)
        token[i] = idx
    mask[: len(seq)] = 0
    mask[np.newaxis, np.newaxis, :]
    return token, mask


VDW_RADIUS = {'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
              'F': 1.47, 'P': 1.8, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98}

def get_vdw_radii(symbol):
    try:
        return VDW_RADIUS[symbol]
    except KeyError:
        return 1.7


def vanderWaals(d, r):
    if d >= 0 and d < r:
        return np.exp(-2 * d * d / (r * r))
    elif d >= r and d < 1.5 * r:
        return max(d * d / (r * r) * 0.541 - d / r * 1.624 + 1.218, 0)
    else:
        return 0


ATOM_TYPES = dict(P=1, C=2, N=3, O=4, F=5, S=6, Cl=7, Br=8, I=9)
HYBRID_TYPES = {
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5
}


# 9 known atom types + 1 unknown type --> 10
# 5 known hybridization types + 1 unknown type --> 6
# degree --> 7
# formal charge --> 1
# radical electrons --> 1
# aromatic states --> 1
# explicit hydrogen --> 5
# chirality --> 3
# dimension == 10 + 6 + 7 + 1 + 1 + 1 + 5 + 3
# dimension == 34
FEATURE_DIM = 34


def get_atom_features(atom):
    
    one_hot = np.zeros(shape=(FEATURE_DIM,), dtype=np.float32)
    
    atom_type_idx = ATOM_TYPES.get(atom.GetSymbol(), 0)
    one_hot[atom_type_idx] = 1
    
    hybrid_type_idx = HYBRID_TYPES.get(atom.GetHybridization(), 0)
    one_hot[10 + hybrid_type_idx] = 1
    
    degree = atom.GetDegree()
    one_hot[16 + degree] = 1
    
    if atom.GetFormalCharge():
        one_hot[23] = 1
    
    if atom.GetNumRadicalElectrons():
        one_hot[24] = 1
        
    if atom.GetIsAromatic():
        one_hot[25] = 1
    
    explicit_h = min(atom.GetTotalNumHs(), 4)
    one_hot[26 + explicit_h] = 1
    
    if atom.HasProp("_ChiralityPossible"):
        one_hot[31] = 1
        try:
            if atom.GetProp('_CIPCode') == 'S':
                one_hot[32] = 1
            else:
                one_hot[33] = 1
        except:
            one_hot[32] = 1
            one_hot[33] = 1
    else:
        one_hot[31:] = 0
    
    return one_hot/sum(one_hot)



#取64为最大原子数  去掉氢原子
def get_mol_features(smiles: str, max_atom_num=DEFAULT_MAX_ATOM_NUM):

    mol = Chem.MolFromSmiles(smiles)
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mol = Chem.RemoveHs(mol)
    
    if mol is None:
        print(f"SMILES {smiles} invalid")
        return None, None, None, None

    atom_num = mol.GetNumAtoms()
    if atom_num > max_atom_num:
        print(f"SMILES {smiles} is too large")
        return None, None, None, None

    x = np.zeros(shape=(max_atom_num, FEATURE_DIM), dtype=np.float32)
    a = np.zeros(shape=(max_atom_num, max_atom_num), dtype=np.float32)

    

    
    atom_rxyz = np.zeros((atom_num, 4), dtype=np.float32)
    
    try:
        mol.GetConformer()
    except ValueError:
        return None, None, None, None

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        x[atom_idx, :] = get_atom_features(atom)
        atom_rxyz[atom_idx, 0] = get_vdw_radii(atom.GetSymbol())
        atom_rxyz[atom_idx, 1:] = mol.GetConformer().GetAtomPosition(atom_idx)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    a_ = Chem.GetAdjacencyMatrix(mol).astype(np.float32) + np.eye(atom_num, dtype=np.float32)
    a_degree = np.diag(np.power(np.sum(a_, axis=1), -0.5))
    a[:atom_num, :atom_num] = np.matmul(np.matmul(a_degree, a_), a_degree)

    
    return x, a



def get_token(smiles, vocab_dict, max_len=100):
    if len(smiles) > max_len:
        return None, None
    
    token = np.zeros(max_len, dtype=np.float32)

    for i, char in enumerate(smiles):
        idx = vocab_dict.get(char, 0)
        token[i] = idx

    return token
#token相当于embedding，重新编码smiles，mask???当前为全零1X1Xlen(smiles)向量


def main_davis(txt_file):
    f = open(txt_file, 'r')
    raw_data = f.read().strip().split('\n')
    f.close()
    
    good_data = []
    for row in raw_data:
        smiles, sequence, label = row.strip().split(',')
        if '.' not in smiles:
            good_data.append([smiles, sequence, label])
    
    f = open("data/davis_vocab.txt")
    vocab_data = f.read().strip().split('\n')
    f.close()
    vocab = {}
    for row in vocab_data:
        char, idx = row.split(',')
        vocab[char] = int(idx)
    print(vocab)
    vocab_size = len(vocab)
    
    
    f = open("data/davis_vocab_protein.txt")
    vocab_data = f.read().strip().split('\n')
    f.close()
    vocab_protein = {}
    for row in vocab_data:
        char, idx = row.split(',')
        vocab_protein[char] = int(idx)
    print(vocab_protein)
    
    
    
    
    
    x_list, a_list= [], []
    protein_list= []
    y_list = []
    token_list = []
    #ecfp_list = []
    smiles_list=[]
    sequence_list=[]
    #good data---- smiles, sequence, label
    total = len(good_data)
    for i, point in enumerate(good_data):
        #if i==100: break
        print(f"{i + 1} / {total}...")
        smiles, sequence, label = point

        x, a = get_mol_features(smiles) #d_mask是1X1X原子数量的全0 向量
        token= get_token(smiles, vocab)  #token 编码后的药物 token_mask 1X1X字符串长度 全零
        mol = Chem.MolFromSmiles(smiles)
        if x is not None and token is not None:
            if smiles not in smiles_list:
                smiles_list.append(smiles)
            if sequence not in sequence_list:
                sequence_list.append(sequence)

            # x_list.append(x)
            # a_list.append(a)

        
            protein_token, p_mask = protein_seq_to_idx_mask(sequence, vocab_protein) #p_mask 1X1X蛋白序列长度
            protein_list.append(protein_token)

            y_list.append(np.array([float(label)]))
            
            token_list.append(token)



            
            # ecfp = np.array(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024).ToBitString()), dtype=np.float32)
            # ecfp_list.append(ecfp)#ecfp编码方式，仅限使用
    for i in range(len(smiles_list)):
        x,a=get_mol_features(smiles_list[i])
        x_list.append(x.tolist())
        a_list.append(a.tolist())

    drug_num=len(smiles_list)
    protein_num=len(sequence_list)
    all_num=drug_num+protein_num


    adj=[[0 for j in range(all_num)] for i in range(all_num)]
    dgl_norm_0=[]
    dgl_norm_1=[]
    dgl_etype=[0 for i in range(total)]
    index=[]
    for i in range(total):
        index.append([])
    for i,point in enumerate(good_data):
        print(f"{i + 1} / {total}...")
        smiles,sequence,label=point
        dgl_norm_0.append(smiles_list.index(smiles))
        dgl_norm_1.append(drug_num + sequence_list.index(sequence))
        index[i].append(smiles_list.index(smiles))
        index[i].append(drug_num + sequence_list.index(sequence))
        l=float(label)
        if l>5.0 and l<6.0:

            dgl_etype[i]=1
            adj[smiles_list.index(smiles)][drug_num+sequence_list.index(sequence)]=1
            adj[drug_num+sequence_list.index(sequence)][smiles_list.index(smiles)]=1

        elif l>=6.0 and l<7.0:
            dgl_etype[i] = 2
            adj[smiles_list.index(smiles)][drug_num + sequence_list.index(sequence)] = 2
            adj[drug_num + sequence_list.index(sequence)][smiles_list.index(smiles)] = 2

        elif l>=7.0 and l<8.0:
            dgl_etype[i] = 3
            adj[smiles_list.index(smiles)][drug_num + sequence_list.index(sequence)] = 3
            adj[drug_num + sequence_list.index(sequence)][smiles_list.index(smiles)] = 3
        elif l>=8.0 and l<9.0:
            dgl_etype[i] = 4
            adj[smiles_list.index(smiles)][drug_num + sequence_list.index(sequence)] = 4
            adj[drug_num + sequence_list.index(sequence)][smiles_list.index(smiles)] = 4
        elif l>=9.0:
            dgl_etype[i] = 5
            adj[smiles_list.index(smiles)][drug_num + sequence_list.index(sequence)] = 5
            adj[drug_num + sequence_list.index(sequence)][smiles_list.index(smiles)] = 5
        else:
            dgl_etype[i]=0
            adj[smiles_list.index(smiles)][drug_num + sequence_list.index(sequence)] = 0
            adj[drug_num + sequence_list.index(sequence)][smiles_list.index(smiles)] = 0

    index=np.array(index,dtype=np.int16)
    dgl0=np.array(dgl_norm_0,dtype=np.int16)
    dgl1=np.array(dgl_norm_1,dtype=np.int16)
    etype=np.array(dgl_etype,dtype=np.int16)
    adj_list=np.array(adj,dtype=np.float32)


    # x_list = np.array(x_list, dtype=np.float32)
    # a_list = np.array(a_list, dtype=np.float32)

    
    protein_list = np.array(protein_list, dtype=np.float32)

    
    y_list = np.array(y_list, dtype=np.float32)
    
    token_list = np.array(token_list, dtype=np.float32)

    print(len(smiles_list))

    filename1 = 'data/davis_kinase.csv_drug_x.json'

    # filename2 = 'kiba.csv_sequence_protein_x.json'
    with open(filename1, 'w') as file_obj1:
        json.dump(x_list, file_obj1)
    filename2 = 'data/davis_kinase.csv_drug_a.json'
    with open(filename2, 'w') as file_obj2:
        json.dump(a_list, file_obj2)
    # np.save(f"davis_kinase.csv_drug_smiles_seq.npy", smiles_list)
    #ecfp_list = np.array(ecfp_list, dtype=np.float32)
    # print(index.shape)
    # print(index)
    # print(dgl0.shape,dgl1.shape)
    # print(etype.shape)
    # print('----------------------')
    # print(x_list.shape, a_list.shape)
    # print(protein_list.shape)
    # print(y_list.shape)
    # print(token_list.shape)
    # print(protein_num)
    # print(drug_num)

    np.save(f"{txt_file}_index.npy",index)
    np.save(f"{txt_file}_dgl0.npy",dgl0)
    np.save(f"{txt_file}_dgl1.npy", dgl1)
    np.save(f"{txt_file}_edg_type.npy", etype)
    np.save(f"{txt_file}_adj.npy", adj_list)
    np.save(f"{txt_file}_x.npy", x_list)
    np.save(f"{txt_file}_a.npy", a_list)
    np.save(f"{txt_file}_protein", protein_list)
    np.save(f"{txt_file}_y.npy", y_list)
    np.save(f"{txt_file}_token.npy", token_list)
    np.save(f"{txt_file}_sequence_list.npy", sequence_list)






    # np.save(f"{txt_file}_ecfp.npy", ecfp_list)




main_davis("data/davis_kinase.csv")
#main_davis("davis_2.csv")
