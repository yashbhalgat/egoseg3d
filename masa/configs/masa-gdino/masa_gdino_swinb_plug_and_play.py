_base_ = [
    '../../projects/grounding_dino/grounding_dino_swin-b_pretrain_mixeddata_masa.py',
    '../default_runtime.py'
]
default_scope = 'mmdet'
detector = _base_.model
detector.pop('data_preprocessor')
detector['init_cfg'] = dict(
    type='Pretrained',
    checkpoint= 'saved_models/tsa_models/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'
    # noqa: E501
)
detector['type'] = 'GroundingDINOMasa'

del _base_.model

model = dict(
    type='MASA',
    freeze_detector=True,
    unified_backbone=True,
    load_public_dets = False,
    given_dets = True,
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        # Image normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        # Image padding parameters
        pad_mask=False,  # In instance segmentation, the mask needs to be padded
        pad_size_divisor=32),  # Padding the image to multiples of 32
    detector=detector,
    masa_adapter=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024],
            out_channels=256,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            num_outs=5),
        dict(
        type='DeformFusion',
         in_channels=256,
        out_channels=256,
        num_blocks=3)],
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.02,
            # nms=dict(type='nms', iou_threshold=0.5),
            nms=dict(type='nms',
                     iou_threshold=0.5,
                     class_agnostic=True,
                     split_thr=100000),
            max_per_img=50,
            mask_thr_binary=0.5)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ),
    track_head=dict(
        type='MasaTrackHead',
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32]),
        embed_head=dict(
            type='QuasiDenseEmbedHead',
            num_convs=4,
            num_fcs=1,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='UnbiasedContrastLoss', loss_weight=0.25),
            loss_track_aux=dict(
                type='MarginL2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0)),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='CombinedSampler',
                num=512,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(type='RandomSampler')))),
    tracker=dict(
        type='MasaTaoTracker',
        init_score_thr=0.1,
        obj_score_thr=0.01,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_momentum=0.8,
        with_cats=False,
        max_distance=100,
        fps=30,
        )
)

inference_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(
                type='Resize',
                scale=(1333, 800),
                keep_ratio=True),
        ]),
    dict(type='PackTrackInputs')
]


# runtime settings
train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    visualization=dict(type='TrackVisualizationHook', draw=False),
checkpoint = dict(type='CheckpointHook', interval=1),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='MasaTrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')

