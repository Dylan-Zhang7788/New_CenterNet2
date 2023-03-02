from detectron2.config import CfgNode as CN

def add_atss_config(cfg):
    _C = cfg
    _C.MODEL.ATSS = CN()
    _C.MODEL.ATSS.NUM_CLASSES = 81  # the number of classes including background

    # Anchor parameter
    _C.MODEL.ATSS.ANCHOR_SIZES = (64, 128, 256, 512, 1024)
    _C.MODEL.ATSS.ASPECT_RATIOS = (1.0,)
    _C.MODEL.ATSS.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
    _C.MODEL.ATSS.STRADDLE_THRESH = 0
    _C.MODEL.ATSS.OCTAVE = 2.0
    _C.MODEL.ATSS.SCALES_PER_OCTAVE = 1

    # Head parameter
    _C.MODEL.ATSS.NUM_CONVS = 4
    _C.MODEL.ATSS.USE_DCN_IN_TOWER = False

    # Focal loss parameter
    _C.MODEL.ATSS.LOSS_ALPHA = 0.25
    _C.MODEL.ATSS.LOSS_GAMMA = 2.0

    # how to select positves: ATSS (Ours) , SSC (FCOS), IoU (RetinaNet), TOPK
    _C.MODEL.ATSS.POSITIVE_TYPE = 'ATSS'

    # IoU parameter to select positves
    _C.MODEL.ATSS.FG_IOU_THRESHOLD = 0.5
    _C.MODEL.ATSS.BG_IOU_THRESHOLD = 0.4

    # topk for selecting candidate positive samples from each level
    _C.MODEL.ATSS.TOPK = 9

    # regressing from a box ('BOX') or a point ('POINT')
    _C.MODEL.ATSS.REGRESSION_TYPE = 'BOX'

    # Weight for bbox_regression loss
    _C.MODEL.ATSS.REG_LOSS_WEIGHT = 2.0

    # Inference parameter
    _C.MODEL.ATSS.PRIOR_PROB = 0.01
    _C.MODEL.ATSS.INFERENCE_TH = 0.05
    _C.MODEL.ATSS.NMS_TH = 0.6
    _C.MODEL.ATSS.PRE_NMS_TOP_N = 1000

    
    # ---------------------------------------------------------------------------- #
    # Specific test options
    # ---------------------------------------------------------------------------- #
    _C.TEST.EXPECTED_RESULTS = []
    _C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
    # Number of images per batch
    # This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
    # see 2 images per batch
    _C.TEST.IMS_PER_BATCH = 8
    # Number of detections per image
    _C.TEST.DETECTIONS_PER_IMG = 100


    # ---------------------------------------------------------------------------- #
    # Test-time augmentations for bounding box detection
    # See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_1x.yaml for an example
    # ---------------------------------------------------------------------------- #
    _C.TEST.BBOX_AUG = CN()

    # Enable test-time augmentation for bounding box detection if True
    _C.TEST.BBOX_AUG.ENABLED = False

    # Horizontal flip at the original scale (id transform)
    _C.TEST.BBOX_AUG.H_FLIP = False

    # Each scale is the pixel size of an image's shortest side
    _C.TEST.BBOX_AUG.SCALES = ()

    # Max pixel size of the longer side
    _C.TEST.BBOX_AUG.MAX_SIZE = 4000

    # Horizontal flip at each scale
    _C.TEST.BBOX_AUG.SCALE_H_FLIP = False

    _C.TEST.BBOX_AUG.VOTE = False
    _C.TEST.BBOX_AUG.VOTE_TH = 0.66
    _C.TEST.BBOX_AUG.SCALE_RANGES = ()
    _C.TEST.BBOX_AUG.MERGE_TYPE = 'vote'
