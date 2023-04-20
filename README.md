
This is a fork of [BEVFusion](https://github.com/ADLab-AutoDrive/BEVFusion), ([paper](https://arxiv.org/abs/2205.13790))


**Installation**

For reference you can check out [getting_started.md](docs/getting_started.md) for installation of mmdet3d.

```shell
conda create -n bevf python=3.8 -y
conda activate bevf
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.11.0
cd bevf
pip install -e . 
```

If required:
```shell
pip install timm
pip install setuptools==59.5.0

```
[data_preparation.md](docs/data_preparation.md) (Run with this codebase, coordinate refactoring issues can occur otherwise)


**Evaluation and Training**
```shell
# training example for bevfusion-pointpillar 
# train nuimage for camera stream backbone and neck.
./tools/dist_train.sh configs/original_bevfusion/cam_stream/mask_rcnn_dbswin-t_fpn_3x_nuim_cocopre.py 8
# first train camera stream
./tools/dist_train.sh configs/original_bevfusion/cam_stream/bevf_pp_4x8_2x_nusc_cam.py 8
# then train LiDAR stream
./tools/dist_train.sh configs/original_bevfusion/lidar_stream/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py 8
# then train BEVFusion
./tools/dist_train.sh configs/original_bevfusion/bevf_pp_2x8_1x_nusc.py 8

# Or for the alt fusion steps:
./tools/dist_train.sh configs/bevfusion_thesis/lr_w_steps/bevf_pp_encodedecode_w_se_misalg_augs.py 8
./tools/dist_train.sh configs/bevfusion_thesis/lr_w_steps/bevf_pp_three_conv_w_se.py 8
#etc

### evaluation example for bevfusion-pointpillar
./tools/dist_test.sh configs/original_bevfusion/bevf_pp_2x8_1x_nusc.py ./work_dirs/bevfusion_pp.pth 8 --eval bbox

```

## Acknowlegement

Thanks the authors of [BEVFusion-Liang](https://github.com/ADLab-AutoDrive/BEVFusion), [BEVFusion-MIT](https://github.com/mit-han-lab/bevfusion), [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [TransFusion](https://github.com/XuyangBai/TransFusion), [LSS](https://github.com/nv-tlabs/lift-splat-shoot), [CenterPoint](https://github.com/tianweiy/CenterPoint).
