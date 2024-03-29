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
import numpy as np
import cv2

from fvcore.common.timer import Timer
from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_setup, launch

from detectron2.evaluation import (
    COCOEvaluator,
    LVISEvaluator,
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
from detectron2.structures import BoxMode

from centernet.config import add_centernet_config
from centernet.data.custom_build_augmentation import build_custom_augmentation

logger = logging.getLogger("detectron2")

def default_argument_parser(epilog=None):
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser = argparse.ArgumentParser(
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="./configs/CenterNet2_R50_1x.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--resume",action="store_true", help="Whether to attempt to resume from the checkpoint directory. ")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")
    parser.add_argument( "--dist-url",default="tcp://127.0.0.1:{}".format(port), help="initialization URL for pytorch distributed backend. See https://pytorch.org/docs/stable/distributed.html for details.")
    parser.add_argument("opts",default=None,nargs=argparse.REMAINDER)
    return parser

def setup(args):  # 根据arg得到cfg的一个函数
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()   # 加载默认的config 
    add_centernet_config(cfg)   # 添加centernet的config
    cfg.merge_from_file(args.config_file)  # 从config_file里合并一部分参数进来
    cfg.merge_from_list(args.opts)  # 自己设置的参数 再通过opt合并进来

    cfg.DATASETS.TRAIN=('balloon_train',)  #训练集
    cfg.DATASETS.TEST=('balloon_val',)  #测试集
    cfg.DATALOADER.NUM_WORKERS=8   #执行序，0是cpu
    cfg.SOLVER.IMS_PER_BATCH=8  #每批次改变的大小
    cfg.SOLVER.BASE_LR=0.01  #学习率
    cfg.SOLVER.STEPS=(4000,8000,)
    cfg.SOLVER.MAX_ITER=20000  #最大迭代次数
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=256  #default:512 批次大小
    # cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    # cfg.SOLVER.CHECKPOINT_PERIOD=5000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES=1  #一类
    cfg.OUTPUT_DIR="./output-balloon"
    
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()   # 冻结参数
    default_setup(cfg, args) # 初始化一下
    return cfg 

def do_test(cfg, model):
    # OrderedDict()是一个有序的词典，Python里的函数
    results = OrderedDict()

    # 进行数据集的转化
    for dataset_name in cfg.DATASETS.TEST:
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' else \
            DatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))
        
        # 加载数据集
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        # 确定评估数据集的类型 lvis或者coco
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type
            
        results[dataset_name] = inference_on_dataset(
            model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)   # solver.build.py
    scheduler = build_lr_scheduler(cfg, optimizer)  # solver.build.py
    
    # 加载一下checkpoint
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    #设置开始的iter数
    start_iter = (
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume,
            ).get("iteration", -1) + 1
    )

    # 如果重置了iter数
    if cfg.SOLVER.RESET_ITER:
        logger.info('Reset loaded iteration. Start training from iteration 0.')
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    # fvcore中的类，用来每一个period存储checkpoint的
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter), # utils.event.py 用来把指标打印到终端的一个东西
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),  # utils.event.py 把指标写进json
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # data.dataset_mapper.py 用来读入数据集，并且完成数据集的裁剪，变形等数据增强工作的
    # build_custom_augmentation是在centernet.data.custom_build_augmentation.py，应该是作者自己写的
    # detectron2里没有这个东西
    mapper = DatasetMapper(cfg, True) if cfg.INPUT.CUSTOM_AUG == '' else \
        DatasetMapper(cfg, True, augmentations=build_custom_augmentation(cfg, True))  
    if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler']:
    # build_detection_train_loader 是在detectron2 的 data.build.py 里定义的
        data_loader = build_detection_train_loader(cfg, mapper=mapper)
    else:
        from centernet.data.custom_dataset_dataloader import  build_custom_train_loader
        data_loader = build_custom_train_loader(cfg, mapper=mapper)


    logger.info("Starting training from iteration {}".format(start_iter))
    # EventStorage 是 detectron2.utils.events.py里面定义的
    # 用来存储训练过程中的指标的
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        
        # 训练一个Batch就是一次Iteration
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)
            
            # loss_dict 是model计算得到的
            # losses是反向传播用的
            # 后面计算的这些都是显示用的
            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)
            
            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()  # 调整lr的一个东西

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                # 定义在detectron2.utils.common.py 里的函数
                # 当使用分布式训练时，在所有进程之间进行同步（屏蔽）的辅助函数
                comm.synchronize()

            if iteration - start_iter > 5 and \
                (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        # 最后记录一下整体的用时
        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))


def get_balloon_dicts(img_dir):
    json_file=os.path.join(img_dir,'via_region_data.json')
    with open(json_file) as f:
        imgs_anns=json.load(f)
    dataset_dicts=[]
    for idx,v in enumerate(imgs_anns.values()):
        record={}  #标准字典档

        filename=os.path.join(img_dir,v['filename'])
        height,width=cv2.imread(filename).shape[:2]  #获取尺寸

        record['file_name']=filename
        
        record['image_id']=idx
        record['height']=height
        record['width']=width

        annos=v['regions']  #范围

        objs=[]
        for _,anno in annos.items():
            assert not anno['region_attributes']
            anno=anno['shape_attributes']
            px=anno['all_points_x']
            py=anno['all_points_y']
            poly=[(x+0.5,y+0.5) for x,y in zip(px,py)] #标记框框
            poly=[p for x in poly for p in x]
            obj={
                'bbox':[np.min(px),np.min(py),np.max(px),np.max(py)], #左上角坐标和右下角坐标
                'bbox_mode':BoxMode.XYXY_ABS,
                'segmentation':[poly],
                'category_id':0, #类别id
                'iscrowd':0    #只有一个类别
            }
            objs.append(obj)
        record['annotations']=objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ['train','val']:  #注册数据集
    DatasetCatalog.register('balloon_'+d,lambda d=d: get_balloon_dicts('datasets/balloon/'+d))
    MetadataCatalog.get('balloon_'+d).set(thing_classes=['balloon'])

def main(args):
    cfg = setup(args) # 前头定义的函数

    model = build_model(cfg)  # modeling.meta_arch.build.py
    logger.info("Model:\n{}".format(model))  # 记录信息的
    if args.eval_only:  # 如果只用于测试
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.TEST.AUG.ENABLED:
            logger.info("Running inference with test-time augmentation ...")
            model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)

        return do_test(cfg, model)  # 如果eval_only=true 那么直接返回do_test

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=False
        )

    do_train(cfg, model, resume=args.resume)  # 上头定义的
    print("finish!!!!!!!!!!!!!!!!")
    return do_test(cfg, model)


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
        args.num_gpus,        
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
