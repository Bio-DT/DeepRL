# DeepRL: a unified multi-level deep representation learning framework to enhance drug discovery and optimization
## Install environment
You can configure the environment by using the following commands:
```
conda create -n DeepRL python=3.9
conda activate DeepRL
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

pip install torch_cluster-1.6.3+pt23cu121-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.2+pt23cu121-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.18+pt23cu121-cp39-cp39-linux_x86_64.whl
pip install torch_spline_conv-1.2.2+pt23cu121-cp39-cp39-linux_x86_64.whl
pip install torch-geometric
Note:Please note: Please uninstall the corresponding.whl file from the official website. https://data.pyg.org/whl/

pip install rdkit= 2024.3.5
pip install biopython=1.83
pip install openbabel=3.1.1 
pip install plip=2.4.0

```
## System requirements
```
We run all of our code on the Linux system. The requirements of this system are as follows:
- Operating System: Ubuntu 22.04.4 LTS
- CPU: Intel® Xeon(R) Platinum 8370C CPU @ 2.80GHz (128GB) 
- GPU: NVIDIACorporationGA100 (A100 SXM480GB)
```

## Data 
```
For DDA prediciton, we use two datasets, including Cdataset and Fdataset. For each dataset, the detailed description is as follows:
- Drug_mol2vec: The mol2vec embeddings for drugs to construct the association network
- DrugFingerprint, DrugGIP: The similarity measurements of drugs to construct the similarity network
- DiseaseFeature: The disease embeddings to construct the association network
- DiseasePS, DiseaseGIP: The similarity measurements of diseases to construct the similarity network
- Protein_ESM: The ESM-2 embeddings for proteins to construct the association network
- DrugDiseaseAssociationNumber: The known drug disease associations
- DrugProteinAssociationNumber: The known drug protein associations
- ProteinDiseaseAssociationNumber: The known disease protein associations

For DTA prediciton, we use two datasets, including Davis and Kiba datasets. For each dataset, the detailed processing methods are as follows:
we use creat_data.py to generate data that conforms to the model's input requirements.

For 3D molecule optimizaiton, we use PDBbind dataset to pretrain model, and the dataset is downloaded from http://www.pdbbind.org.cn/.
Following this, process this dataset by preprocessing.py

```



## Training model
```
We 
```
