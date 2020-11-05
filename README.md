# Point Sampling Net: Fast Subsampling and Local Grouping for Deep Learning on Point Cloud
### (Anonymous Currently)

This repo is implementation for our paper :
*Point Sampling Net: Fast Subsampling and Local Grouping for Deep Learning on Point Cloud*

## Introduction
![Architecture of Point Sampling Net](https://github.com/psn-anonymous/PointSamplingNet/blob/master/image/psn.png "Architecture of Point Sampling Net")
**Point Sampling Net** is a differentiable fast grouping and sampling method for deep learning on point cloud, which can be applied to mainstream point cloud deep learning models. Point Sampling Net perform grouping and sampling tasks at the same time. It does not use the relationship between points as a grouping reference, so that the inference speed is independent of the number of points, and friendly to parallel implementation, that reduces the time consumption of sampling and grouping effectively.

## Environments
This repo has been tested on follow environments
### Software
Ubuntu 20.04<br>
Python 3.8.5<br>
PyTorch 1.7.0<br>
NVIDIA CUDA Toolkit 10.2<br>
NVIDIA cudnn 7.6.5<br>
<br>
You can build the software environment through **conda**  easily
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
### Hardware
Intel Core i9 9900K<br>
NVIDIA TITAN RTX

## Classification
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

### Run
```
python train_cls.py --log_dir [your log dir]
```

## Part Segmentation
### Data Preparation
Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)  and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`.
### Run
```
python train_partseg.py --normal --log_dir [your log dir]
```

## Semantic Segmentation
### Data Preparation
Download 3D indoor parsing dataset (**S3DIS**) [here](http://buildingparser.stanford.edu/dataset.html)  and save in `data/Stanford3dDataset_v1.2_Aligned_Version/`.
```
cd data_utils
python collect_indoor3d_data.py
```
Processed data will save in `data/stanford_indoor3d/`.
### Run
```
python train_semseg.py --log_dir [your log dir]
python test_semseg.py --log_dir [your log dir] --test_area 5 --visual
```


## Reference
Our experiment is heavily referenced [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)<br>
Thanks!
