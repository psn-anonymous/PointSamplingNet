# Point Sampling Net: Fast Subsampling and Local Grouping for Deep Learning on Point Cloud
### (Anonymous Currently)

This repository is the implementation for our paper :<br>
*Point Sampling Net: Fast Subsampling and Local Grouping for Deep Learning on Point Cloud*

## Introduction
![Architecture of Point Sampling Net](https://github.com/psn-anonymous/PointSamplingNet/blob/master/image/psn.png "Architecture of Point Sampling Net")<br>
**Point Sampling Net** is a differentiable fast grouping and sampling method for deep learning on point cloud, which can be applied to mainstream point cloud deep learning models. Point Sampling Net perform grouping and sampling tasks at the same time. It does not use the relationship between points as a grouping reference, so that the inference speed is independent of the number of points, and friendly to parallel implementation, that reduces the time consumption of sampling and grouping effectively.<br>
Point Sampling Net has been tested on PointNet++, PointConv, RS-CNN, GAC. There is not obvious adverse effects on these deep learning models of classification, part segmentation, and scene segmentation tasks and the speed of training and inference has been significantly improved.



# Usage
The [**CORE FILE**](https://github.com/psn-anonymous/PointSamplingNet/blob/master/models/PointSamplingNet.py) of Point Sampling Net: [models/PointSamplingNet.py](https://github.com/psn-anonymous/PointSamplingNet/blob/master/models/PointSamplingNet.py)

## Software Dependencies
Python 3.7 or newer<br>
PyTorch 1.5 or newer<br>
NVIDIA® CUDA® Toolkit 9.2 or newer<br>
NVIDIA® CUDA® Deep Neural Network library (cuDNN) 7.2 or newer<br>
<br>
You can build the software dependencies through **conda**  easily
```
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
```

## Import Point Sampling Net PyTorch Module
You may import PSN pytorch module by:
```
import PointSmaplingNet as psn
```
## Native PSN
### Defining
```
psn_layer = psn.PointSamplingNet(num_to_sample = 512, max_local_num = 32, mlp = [32, 256])
```
Attribute *mlp* is the middle channels of PSN, because the channel of first layer and last layer must be 3 and sampling number.
### Forward Propagation
```
sampled_indices, grouped_indices = psn_layer(coordinate = %coordinates of point cloud%)
```
*sampled_indices* is the indices of sampled points, *grouped_indices* is the grouped indices of points.<br>
*%coordinates of point cloud%* is a torch.Tensor object, its shape is [*batch size*, *number of points*, *3*].

## PSN with Heuristic Condition
An example of PSN with radius query.
### Defining
```
psn_radius_layer = psn.PointSamplingNetRadius(num_to_sample = 512, radius = 0.2, max_local_num = 32, mlp = [32, 256])
```
### Forward Propagation
```
sampled_indices, grouped_indices = psn_radius_layer(coordinate = %coordinates of point cloud%)
```
*sampled_indices* is the indices of sampled points, *grouped_indices* is the grouped indices of points.<br><br>
You may implement your own heuristic condition function C(x) and replace the radius query function.<br><br>
*Warning : We strongly recommend that you do **NOT** use heuristic condition if it is not necessary, because it may reduce the number of local meaningful features.*

## PSN with Multi-Scale Grouping
### Defining
```
psn_msg_layer = psn.PointSamplingNetMSG(num_to_sample = 512, msg_n = [32, 64], mlp = [32, 256])
```
Attribute *msg_n* is the list of multi-scale *n*.
### Forward Propagation
```
sampled_indices, grouped_indices_msg = psn_msg_layer(coordinate = %coordinates of point cloud%)
```
*sampled_indices* is the indices of sampled points, *grouped_indices_msg* is grouped indices of points of list of mutil-scale.


# Visualize Effect
## Sampling
![Visualize of Sampling](https://github.com/psn-anonymous/PointSamplingNet/blob/master/image/plane1.png "Visualize of Sampling")
## Grouping
![Visualize of Grouping](https://github.com/psn-anonymous/PointSamplingNet/blob/master/image/plane2.png "Visualize of Grouping")

# The Experiment on Deep Learning Networks
There is an experiment on PointNet++
## Environments
This experiment has been tested on follow environments:
### Software
Canonical Ubuntu 20.04.1 LTS / Microsoft Windows 10 Pro<br>
Python 3.8.5<br>
PyTorch 1.7.0<br>
NVIDIA® CUDA® Toolkit 10.2.89<br>
NVIDIA® CUDA® Deep Neural Network library (cuDNN) 7.6.5<br>

### Hardware
Intel® Core™ i9-9900K Processor (16M Cache, up to 5.00 GHz)<br>
64GB DDR4 RAM<br>
NVIDIA® TITAN RTX™

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
This implementation of experiment is heavily reference to [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)<br>
Thanks very much !
