import logging
import os
from collections import OrderedDict
from pickle import TRUE
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime
import json
import argparse
import sys

from torch.utils.tensorboard import SummaryWriter

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_setup, launch

from detectron2.evaluation import (
    COCOEvaluator,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader

from centernet.config import add_centernet_config
from centernet.data.custom_build_augmentation import build_custom_augmentation

from centernet.MY_evaluation.MY_coco_evaluation import MY_COCOEvaluator
from centernet.MY_evaluation.MY_pascal_voc_evaluation import MY_PascalVOCDetectionEvaluator
from centernet.MY_datasets.MY_pascal_voc import MY_register
from centernet.MY_datasets.MY_balloon import MY_register_balloon

logger = logging.getLogger("detectron2")
MY_register()
MY_register_balloon()
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def default_argument_parser(epilog=None):
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser = argparse.ArgumentParser(
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="configs/My_CenterNet-BiFPN.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--resume",action="store_true",help="Whether to attempt to resume from the checkpoint directory.",)
    parser.add_argument("--eval-only",action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=2, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")
    parser.add_argument("--dist-uputrl",default="tcp://127.0.0.1:{}".format(port),help="initialization URL for pytorch distributed backend. See""https://pytorch.org/docs/stable/distributed.html for details.",)
    parser.add_argument("--find_unused_parameters",default=True)
    parser.add_argument("--start_eval_period",type=int,default=0)
    parser.add_argument("opts",default=None,nargs=argparse.REMAINDER,)

def setup(args):  # 根据arg得到cfg的一个函数
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()   # 加载默认的config 
    add_centernet_config(cfg)   # 添加centernet的config
    cfg.merge_from_file(args.config_file)  # 从config_file里合并一部分参数进来
    cfg.merge_from_list(args.opts)  # 自己设置的参数 再通过opt合并进来
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    # cfg.MODEL.WEIGHTS="./models/CenterNet2_R50_1x.pth"
    # cfg.DATALOADER.NUM_WORKERS=8   #执行序，0是cpu
    # cfg.SOLVER.IMS_PER_BATCH=16  #每批次改变的大小
    # cfg.SOLVER.BASE_LR=0.01  #学习率
    # cfg.SOLVER.STEPS=(60000,80000,)
    # cfg.SOLVER.MAX_ITER=120000  #最大迭代次数
    # cfg.SOLVER.CHECKPOINT_PERIOD=40
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=256  #default:512 批次大小
    cfg.freeze()   # 冻结参数
    default_setup(cfg, args) # 初始化一下
    return cfg

def main(args):
    cfg = setup(args) # 前头定义的函数
    mapper = DatasetMapper(cfg, True) if cfg.INPUT.CUSTOM_AUG == '' else \
    DatasetMapper(cfg, True, augmentations=build_custom_augmentation(cfg, True))
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    for data, iteration in zip(data_loader, range(start_iter, max_iter)):

if __name__ == "__main__":
    args = default_argument_parser() # engine.default.py
    args.add_argument('--manual_device', default='') # python 
    args = args.parse_args() #python 
    if args.manual_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.manual_device  # 指定训练的显卡
    args.dist_url = 'tcp://127.0.0.1:{}'.format(
        torch.randint(11111, 60000, (1,))[0].item())   # 分布式训练的什么东西 不懂
    print("Command Line Args:", args)    # engine.launch 第一个是函数，后面全是参数
    launch(
        main,
        # 后面的这些参数全都在 default_argument_parser() 里头定义的 
        # 要改可以 args.xxx=xxx 就行了
        args.num_gpus,        
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )