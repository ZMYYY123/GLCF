import torch
import esm
import json
import numpy as np
from collections import OrderedDict

model, alphabet = esm.pretrained.esm2_t24_35M_UR50D()
batch_converter = alphabet.get_batch_converter()

proteins = np.load(f"davis_list.npy",allow_pickle=True)


tuple_list = [tuple((index,item)) for index,item in enumerate(proteins)]


batch_labels, batch_strs, batch_tokens = batch_converter(tuple_list)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
token_representations=[]
for i, tokens_len in enumerate(batch_tokens):
    t=batch_tokens[i]
    with torch.no_grad():
        results = model(t, repr_layers=[33], return_contacts=False)
    token_representations.append(results["representations"][33])


token_representations=torch.stack()
token_representations=torch.squeeze()

sequence_representations = token_representations.numpy()
print(sequence_representations.shape)
np.save(f"davis_mutilable/davis_protein.npy", sequence_representations)



