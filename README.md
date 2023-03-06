
This is a fork of [paper](https://arxiv.org/abs/2205.13790) 


**Installation**

Please refer to [getting_started.md](docs/getting_started.md) for installation of mmdet3d.

Recommended environments:

```shell
python==3.8.3
mmdet==2.11.0 (please install mmdet in mmdetection-2.11.0)
mmcv==1.4.0
mmdet3d==0.11.0
numpy==1.19.2
torch==1.7.0
torchvision==0.8.0
```

**Benchmark Evaluation and Training**

Please refer to [data_preparation.md](docs/data_preparation.md) to prepare the data. Then follow the instruction there to train our model. All detection configurations are included in [configs](configs/).

```shell
# training example for bevfusion-pointpillar 
# train nuimage for camera stream backbone and neck.
./tools/dist_train.sh configs/bevfusion/cam_stream/mask_rcnn_dbswin-t_fpn_3x_nuim_cocopre.py 8
# first train camera stream
./tools/dist_train.sh configs/bevfusion/cam_stream/bevf_pp_4x8_2x_nusc_cam.py 8
# then train LiDAR stream
./tools/dist_train.sh configs/bevfusion/lidar_stream/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py 8
# then train BEVFusion
./tools/dist_train.sh configs/bevfusion/bevf_pp_2x8_1x_nusc.py 8

### evaluation example for bevfusion-pointpillar
./tools/dist_test.sh configs/bevfusion/bevf_pp_2x8_1x_nusc.py ./work_dirs/bevfusion_pp.pth 8 --eval bbox

### training example for sota Results with BEV_augs
git checkout dev_aug
# first train camera stream
./tools/dist_train.sh configs/bevfusion/cam_stream/bevf_tf_4x8_20e_nusc_cam_lr.py 8
# then train LiDAR stream
./tools/dist_train.sh configs/bevfusion/lidar_stream/transfusion_nusc_voxel_L.py 8
# then train BEVFusion
./tools/dist_train.sh configs/bevfusion/bevf_tf_4x8_10e_nusc_aug.py 8


```

## Acknowlegement

We sincerely thank the authors of [BEVFusion-Liang](https://github.com/ADLab-AutoDrive/BEVFusion), [BEVFusion-MIT](https://github.com/mit-han-lab/bevfusion), [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [TransFusion](https://github.com/XuyangBai/TransFusion), [LSS](https://github.com/nv-tlabs/lift-splat-shoot), [CenterPoint](https://github.com/tianweiy/CenterPoint).
