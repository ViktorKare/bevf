_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/nuim_instance.py',
    '../../_base_/schedules/mmdet_schedule_1x.py', '../../_base_/default_runtime.py'
]
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
total_epochs = 36

#pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'

model = dict(
    #pretrained = pretrained,  
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch'
        ),
    neck=dict(
        in_channels=[256, 512, 1024, 2048]),
    roi_head=dict(
        bbox_head=dict(num_classes=10), mask_head=dict(num_classes=10)))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)
# fp16 = dict(loss_scale=32.0)

#load_from = 'work_dirs/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'