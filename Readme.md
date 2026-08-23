# GLCF: A Global-Local Collaborative Fusion Framework for Drug-Target Affinity Prediction

## Requirements
```shell
pip install fair-esm
rdkit-pypi
transformers>=4.20.0
```

## Datasets

/main/data/davis_kinase.csv
/main/GLCF/data/kiba.csv
/main/GLCF/data/PDBbind


## Usage
* Data Preprocessing:
```python
python preprocess_davis.py 
python preprocess_kiba.py 
```

* Protein graph construction:
```python
python davis_protein_process.py
python kiba_protein_process.py
```
* sequence embeddings:
```python
python drug_embedding.py  
python protein_embedding.py
```
* Train RGCN on the heterogeneous graph
The RGCN learns node embeddings for both drugs and proteins from the interaction graph.
