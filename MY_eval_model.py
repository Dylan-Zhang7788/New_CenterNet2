from train_net import default_argument_parser, do_test, setup, build_model
from collections import OrderedDict
import os
import torch
import time
from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.evaluation import (
    COCOEvaluator,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.data.dataset_mapper import DatasetMapper
from fvcore.common.timer import Timer
from centernet.data.custom_build_augmentation import build_custom_augmentation
from centernet.MY_evaluation.MY_coco_evaluation import MY_COCOEvaluator
from centernet.MY_evaluation.MY_pascal_voc_evaluation import MY_PascalVOCDetectionEvaluator
from centernet.MY_datasets.MY_pascal_voc import MY_register

def main(args):
    cfg = setup(args) # 前头定义的函数
    # MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    # MODEL.DEVICE = "cuda"

    # 这里调用了 META_ARCH_REGISTRY.get()(cfg)
    # 并且使用了注册器，注册了"GeneralizedRCNN"
    # "GeneralizedRCNN"这个类被写在
    model = build_model(cfg)  # modeling.meta_arch.build.py
    dir=cfg.OUTPUT_DIR + "-eval-result/"
    start_iter=0
    max_iter = cfg.SOLVER.MAX_ITER
    writers = (
        [ 
            CommonMetricPrinter(max_iter),
            TensorboardXWriter(dir),
            JSONWriter(os.path.join(dir, "metrics.json")),  # utils.event.py 把指标写进json
        ]
    )
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
            dir, "inference_{}".format(dataset_name))
        # 确定评估数据集的类型 lvis或者coco
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            # evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
            evaluator = MY_COCOEvaluator(dataset_name, cfg, True, output_folder,Writer=None)
        elif evaluator_type == 'MY_pascal_voc':
            evaluator = MY_PascalVOCDetectionEvaluator(dataset_name,Writer=None)

        else:
            assert 0, evaluator_type

    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        storage.step()
        f_list = os.listdir(cfg.OUTPUT_DIR)
        for file in f_list[:]:
            if os.path.splitext(file)[1] != '.pth': 
                f_list.remove(file)

        f_list.sort()
        # print f_list
        for i, file in enumerate(f_list):
            # os.path.splitext():分离文件名与扩展名
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            os.path.join(cfg.OUTPUT_DIR,file), resume=args.resume)
            if cfg.TEST.AUG.ENABLED:
                model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)
            storage.step()
            evaluator.Writer=storage
            results[dataset_name] = inference_on_dataset(
                model, data_loader, evaluator)
            for writer in writers:
                writer.write()
                

if __name__=='__main__':

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