# 3D-Spatial-Grounding

## Environments
- Python 3.8
- CUDA Version 11.8

```
pip install -r requirements.txt
```

### Installing PointNet++
```
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install ./pointnet2_ops_lib
```
Or if you would like to install them directly - however, this didn't work well to me so I recommend following the script above. (cloning the pointnet repository)
```
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## Download pretrained Uni3D and EVA-CLIP
1. Download pretrained Uni3D-B from the following link and place it put them in `models/Uni3D/path/to/checkpoints` folder : [Uni3D](https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-b/model.pt)
2. [Recommended 🤗] Download the [clip](https://huggingface.co/timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k/blob/main/open_clip_pytorch_model.bin) model and put it in `models/Uni3D/path/to/clip_model` folder.

## Login to wandb for logging training metrics
```
wandb init
```

## Train model
```
cd scripts
chmod 755 train.sh
./train.sh
```