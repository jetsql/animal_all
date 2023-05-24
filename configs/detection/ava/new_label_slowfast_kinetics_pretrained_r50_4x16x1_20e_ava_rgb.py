# model setting
#更改參考https://blog.csdn.net/lanyan90/article/details/125796563?ops_request_misc=&request_id=&biz_id=102&utm_term=slowfast%20%E8%A8%93%E7%B7%B4&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-125796563.nonecase
#bash tools/dist_train.sh configs/detection/ava/new_label_slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py 2
model = dict(
    type='FastRCNN',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        # 对应论文中的参数τ
        #0421設定是8，0426設定16
        #論文設定是16，我改良成8
        resample_rate=8, #改
        # 对应论文中的参数α
        speed_ratio=8,
        # 其对应论文中β的倒数
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(1, 2, 2, 1))),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            in_channels=2304,
            #全部類別有幾個，看data/ava/annotations/labelmap.txt
            num_classes=6,#改，原本類別數＋1
            #最後輸出要顯示幾個類別，6類
            # top_k={1,6}, #新增，0506有bug待解決
            multilabel=True,
            dropout_ratio=0.5)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.7, #改，原0.9
                neg_iou_thr=0.7, #改，原0.9
                min_pos_iou=0.7), #改，原0.9
            sampler=dict(
                type='RandomSampler',
                num=8,#0522改，原本32
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0,
            debug=False)),
    test_cfg=dict(rcnn=dict(action_thr=0.002)))

dataset_type = 'AVADataset'
data_root = 'data/ava/rawframes'
anno_root = 'data/ava/annotations'

ann_file_train = f'{anno_root}/ava_train_v2.1.csv'
ann_file_val = f'{anno_root}/ava_val_v2.1.csv'

exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.1.csv'
exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.1.csv'

label_file = f'{anno_root}/ava_action_list_v2.1_for_activitynet_2018.pbtxt'

proposal_file_train = (f'{anno_root}/ava_dense_proposals_train.FAIR.'
                       'recall_93.9.pkl')
proposal_file_val = f'{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key=['proposals', 'gt_bboxes', 'gt_labels'], stack=False)
        ]),
    dict(
        type='Collect',
        keys=['img', 'proposals', 'gt_bboxes', 'gt_labels'],
        meta_keys=['scores', 'entity_ids'])
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals']),
    dict(type='ToDataContainer', fields=[dict(key='proposals', stack=False)]),
    dict(
        type='Collect',
        keys=['img', 'proposals'],
        meta_keys=['scores', 'img_shape'],
        nested=True)
]

data = dict(
    #單卡要把gpu改1
    videos_per_gpu=1,
    workers_per_gpu=1,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        person_det_score_thr=0.7, #改，原0.9
        #自己新增起始編號跟結束編號
        # timestamp_start=2,
        # timestamp_end=27030,
        #分類的類別數+1
        num_classes=6, #新增
        #rawframes編號從幾號開始編
        start_index=1, #新增
        data_prefix=data_root),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        #原本預設0.9
        person_det_score_thr=0.7,
        #自己新增起始編號跟結束編號
        # timestamp_start=2,
        # timestamp_end=27030,
        #分類的類別數+1
        num_classes=6, #新增
        #rawframes編號從幾號開始編
        start_index=1, #新增
        data_prefix=data_root))
data['test'] = data['val']

optimizer = dict(type='SGD', lr=0.1125, momentum=0.9, weight_decay=0.00001)
# this lr is used for 8 gpus

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy

lr_config = dict(
    policy='step',
    step=[10, 15],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1)
total_epochs = 200
# total_epochs = 1
#5個以後存一次權重
checkpoint_config = dict(interval=50)
#運行 1 個 epoch 進行訓練
workflow = [('train', 1)] #原版
#5個以後validation一次
evaluation = dict(interval=20, save_best='mAP@0.5IOU')
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = ('./work_dirs/ava/'
#             'slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb')
#跑多卡GPU路徑要在這邊設定
work_dir = ('./tests/0520_nl_true_ew2/')
# load_from = ('https://download.openmmlab.com/mmaction/recognition/slowfast/'
#              'slowfast_r50_4x16x1_256e_kinetics400_rgb/'
#              'slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth')
load_from=('./checkpoints/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-6e7c704d.pth')
resume_from = None
find_unused_parameters = False
