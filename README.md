# DeepRL: a unified multi-level deep representation learning framework to enhance drug discovery and optimization
## Install environment
You can install the required packages by running the following commands:
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

pip install rdkit 
pip install biopython
pip install openbabel
pip install plip

```

