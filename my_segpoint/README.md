# Reproduction of SegPoint (He, ECCV2024)

## Environments
- Python 3.9
- CUDA Version 11.8

```
conda create -n segpoint python=3.9
conda install -c conda-forge compilers
conda install nvidia/label/cuda-11.8.0::cuda-toolkit -c nvidia/label/cuda-11.8.0
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
```

### Installing PointNet++
```
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install ./pointnet2_ops_lib
```
Or if you want to install them directly (cloning the pointnet repository)
```
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## Download pretrained Uni3D and EVA-CLIP
1. Download pretrained Uni3D-B from the following link and put them in `model/modules/Uni3D/path/to/checkpoints` folder : [Uni3D](https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-b/model.pt)

## Install KPConv libraries


## Login to wandb for logging training metrics
```
wandb init
```

## Train model
```
cd scripts
chmod 755 train.sh
cd ..
./scripts/train.sh
```