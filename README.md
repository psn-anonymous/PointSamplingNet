# Point Sampling Net: Fast Subsampling and Local Grouping for Deep Learning on Point Cloud 

This repo is implementation for our paper:<br>
Point Sampling Net: Fast Subsampling and Local Grouping for Deep Learning on Point Cloud


## Classification
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

### Run
```
## Check model in ./models 

python train_cls.py --model pointnet2_cls_ssg_psn --normal --log_dir pointnet2_cls_ssg_psn
python test_cls.py --normal --log_dir pointnet2_cls_ssg_psn
```

## Part Segmentation
### Data Preparation
Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)  and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`.
### Run
```
## Check model in ./models 

python train_partseg.py --model pointnet2_part_seg_ssg_psn --normal --log_dir pointnet2_part_seg_ssg_psn
python test_partseg.py --normal --log_dir pointnet2_part_seg_ssg_psn
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
## Check model in ./models 
## E.g. pointnet2_ssg
python train_semseg.py --model pointnet2_sem_seg_psn --test_area 5 --log_dir pointnet2_sem_seg_psn
python test_semseg.py --log_dir pointnet2_sem_seg_psn --test_area 5 --visual
```


## Reference
Our experiment is heavily referenced
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)<br>
Thanks!


## Environments
Ubuntu 20.04 <br>
Python 3.8.5 <br>
Pytorch 1.7.0
