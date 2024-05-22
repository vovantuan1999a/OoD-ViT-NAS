# OoD-ViT-NAS: Vision Transformer Neural Architecture Search for Out-of-Distribution Generalization: Benchmark and Insights

# Installation
You use conda to create a virtual environment to run this project.

```bash

cd  OoD-ViT-NAS
conda create --name OoDViTNAS python=3.6.9
conda activate OoDViTNAS
pip install -r requirements.txt
```
After this, you should installl pytorch and torchvision package which meet your GPU and CUDA version according to https://pytorch.org
And install nvidia dail according to https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html
	
Quick Start
=====================================

Dataset
--------------
You can download the dataset from: [Link Datasets].

The dataset is divided into three AutoFormer search spaces: Supernet-Base, Supernet-Small, and Supernet-Tiny. Each dataset includes evaluation results (ID Accuracy, OoD Accuracy) for 8 large common OoD datasets: ImageNet-C, ImageNet-A, ImageNet-O, ImageNet-P, ImageNet-D, ImageNet-R, ImageNet-Sketch, and Stylized ImageNet. This structure helps keep file sizes manageable and allows for selective evaluation results. To use the provided helper class, you need to download the merged JSON data files: OoD-ViT-NAS-Small.json, OoD-ViT-NAS-Tiny.json, and OoD-ViT-NAS-Base.json.

Evaluation of ViT AutoFormer Pipeline on 8 Common OoD Datasets
--------------
If you want to use this project to evaluate ViT architectures from the AutoFormer supernet (https://github.com/microsoft/Cream/tree/main/AutoFormer), you can create a work directory for saving configuration files, JSON results, etc.

We provide an example of evaluating ViT architectures from the AutoFormer supernet-small, supernet-tiny, and supernet-base on 8 common OoD datasets: ImageNet-C, ImageNet-A, ImageNet-O, ImageNet-P, ImageNet-D, ImageNet-R, ImageNet-Sketch, and Stylized ImageNet.

1.Download the pretrained weight models for AutoFormer supernet-small, supernet-tiny, and supernet-base from AutoFormer on GitHub: https://github.com/microsoft/Cream/tree/main/AutoForme and place it into the specified directory.

    mkdir supernets_checkpoints
    cd supernets_checkpoints
    # Then, place the pretrained weight models for supernet-small, supernet-tiny, and supernet-base into this directory.

    mkdir output_OoD_eval_json
    # The directory to save the evaluation results of ViT architectures from the AutoFormer search space on OoD datasets in JSON format files.


2.Edit the configuration file for the OoD datasets located in the './configs' directory

```python
├── configs
│   ├── ImageNet-A.yaml
│   ├── ImageNet-C.yaml
│   ├── ImageNet-D.yaml
│   ├── ImageNet-O.yaml
│   ├── ImageNet-P.yaml
│   ├── ImageNet-R.yaml
│   ├── ImageNet-Sketch.yaml
│   └── ImageNet-Stylized.yaml
```

```python	

# change the dataset path for ImagetNet-O (ImageNet-O.yaml)
data_set: ImageNet-O
imagenet_root_dir:  /Your/Path/Of/ImageNet-C
imagetnet_original_dir: /Your/Path/Of/ImageNet-1k #original imagenet-1k
type_corruption: imagetnet-O
level_corruption: None

# change the dataset path for ImagetNet-C (ImageNet-C.yaml)
data_set: ImageNet-C
imagenet_root_dir:  /Your/Path/Of/ImageNet-C
type_corruption: Gaussian Noise #or other corruption of ImagetNet C (Pixelate, Fog, Snow, ...)
level_corruption: '5' #or other level corruption of ImagetNet C (1, 2, 3, 4)

# change the dataset path for ImagetNet-D (ImageNet-D.yaml)
data_set: ImageNet-D
imagenet_root_dir:  .  /Your/Path/Of/ImageNet_D
type_corruption: texture #other type of ImagetNet D (background, material)
level_corruption: nonehttps://www.overleaf.com/8627546856fjyhbgtmtbvw#5aa6ab

# change the dataset path for ImagetNet-Stylized (ImageNet-Stylized.yaml)
data_set: ImageNet-Stylized
imagenet_root_dir:  ./Your/Path/Of/ImageNet_Stylized
type_corruption: stylized-imagenet
level_corruption: None
```


3.Edit evaluation_OoD_ViT_AutoFM.sh to specify the paths for saving the OoD dataset configuration and the ViT architectures from the AutoFormer supernet search space that you want to evaluate.

```python
python -m torch.distributed.launch --nproc_per_node=2 --use_env evolution_OoD_Vit_AutoFM.py --gp \
--change_qk --relative_position --dist-eval --search-space ./experiments/supernet/supernet-S.yaml --supernet 'small' \
--config-dataset './configs/ImageNet-C.yaml'

python -m torch.distributed.launch --nproc_per_node=2 --use_env evolution_OoD_Vit_AutoFM.py --gp \
--change_qk --relative_position --dist-eval --search-space ./experiments/supernet/supernet-S.yaml --supernet 'small' \
--config-dataset './configs/ImageNet-Stylized.yaml'

python -m torch.distributed.launch --nproc_per_node=2 --use_env evolution_OoD_Vit_AutoFM.py --gp \
--change_qk --relative_position --dist-eval --search-space ./experiments/supernet/supernet-B.yaml --supernet 'base' \
--config-dataset './configs/ImageNet-D.yaml'

python -m torch.distributed.launch --nproc_per_node=2 --use_env evolution_OoD_Vit_AutoFM.py --gp \
--change_qk --relative_position --dist-eval --search-space ./experiments/supernet/supernet-T.yaml --supernet 'base' \
--config-dataset './configs/ImageNet-O.yaml'
```

4.Run it

    bash evaluation_OoD_Vit_AutoFM.sh

Training-free NAS pipeline for OoD Generalization
--------------

1.Edit compute_training_free_for_ViT_OoD.sh to specify the paths for your OoD datasets and the ViT architectures from the AutoFormer supernet search space that you want to evaluate.

```python
python compute_9_trainning_frees_for_Vit_OoD.py --data-path /Your/Path/Of/ImageNet_OoD --gp \
 --change_qk --relative_position --dist-eval --search-space './experiments/supernet/supernet-B.yaml' --output_dir './OUTPUT/trainning_free_nas' --supernet 'small'
```

2.Run it

    bash compute_9_trainning_frees_for_Vit_OoD.sh


Usage
-------
This repository contains a helper class, ViTOoD_dataset.py, to access the data. Below is an example demonstrating how to use this helper class.

We provide API access to JSON data, enabling you to evaluate Out-of-Distribution (OoD) performance results on Vision Transformer (ViT) architectures. These architectures are derived from the AutoFormer search space and tested across 8 and commonly used OoD datasets: ImageNet-A, ImageNet-O, ImageNet-P, ImageNet-C, ImageNet-D, ImageNet-R, Stylized ImageNet, and ImageNet-Sketch.


```python
from ViTOoD_dataset import VitOoDDataset
data = VitOoDDataset(path=path_to_merge_json_data_file_OoD_Vit_NAS)

results = data.query(
    # data specifies the evaluated dataset
    data = ["imagenet-D", "imagenet-C", "imagenet-A","stylized-imagenet","sketch-imagenet"],
    # measure specifies the evaluation type
    measure = "OoD performance",
    # key specifies the corruption types
    key = VitOoDDataset.keys_all,
    level = "5", #only for IN-C (IN-C have 5 level corruption) (int: 1,2,3,4 or 5)
)
print(results['imagenet-C']['728']['net_setting'])
print(results['imagenet-A']['728']["Gaussian Noise"])
print(results['imagenet-D']['728']["texture"])
print(results['imagenet-sketch']['728'])['sketch']
print(results['stylized-imagenet']['728'])['stylized-imagenet']
```
