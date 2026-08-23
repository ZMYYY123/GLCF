import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer,T5Model

import numpy as np
encoder = AutoModelForMaskedLM.from_pretrained("ChemBERTa-77M-MLM")

tokenizer = AutoTokenizer.from_pretrained("ChemBERTa-77M-MLM")


drugs = np.load(f"davis.npy",allow_pickle=True)
drugs=list(drugs)



batch_size = 10
total_samples = len(drugs)
num_batches = (total_samples + batch_size - 1) // batch_size
result=[]
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, total_samples)

    batch_sentences = drugs[start_idx:end_idx]


    inputs = tokenizer(batch_sentences,padding=True, max_length=200, truncation=True, return_tensors="pt")
    outputs = encoder(**inputs)
    logits = outputs.logits

    for j in range(len(batch_sentences)):

        result.append(logits[j])
res=torch.stack(result)
result=res.detach().numpy()
print(result.shape)
