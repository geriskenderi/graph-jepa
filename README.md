# Graph-JEPA

### Paper and citation

[Graph-level Representation Learning with Joint-Embedding Predictive Architectures](https://arxiv.org/abs/2309.16014)

```
@article{skenderi2023graph,
  title={Graph-level Representation Learning with Joint-Embedding Predictive Architectures},
  author={Skenderi, Geri and Li, Hang and Tang, Jiliang and Cristani, Marco},
  journal={arXiv preprint arXiv:2309.16014},
  year={2023}
}
```

### Python environment setup with Conda

```
conda create --name graphjepa python=3.8
conda activate graphjepa
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install yacs
pip install tensorboard
pip install networkx
pip install einops
pip install metis
```

### Run Graph-JEPA

The code for all the available datasets used in the paper is available under the `train/` folder. Inside `train/configs` you can find the specific configuration files used for each dataset, corresponding with Table 4 of the paper.

`launch.sh` contains two examples of launch scripts that you can use to directly modify the default config. You can see a detailed list of the arguments in `core/config.py`

### Reproducibility

We provide the training logs containing the results published in the paper in `paper_logs/`

### Credits
This repository is largely based on [Graph-ViT-MLPMixer](https://github.com/XiaoxinHe/Graph-ViT-MLPMixer), check out their work as well if you are interested in graph-level tasks.
