import numpy as np
import json
import cv2
import os
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog  #注册Metadata
from detectron2.data import DatasetCatalog   #注册资料集
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode  #标记方式
from matplotlib import pyplot as plt
from centernet.config import add_centernet_config

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
    DatasetCatalog.register('balloon_'+d,lambda d=d: get_balloon_dicts('./datasets/balloon/'+d))
    MetadataCatalog.get('balloon_'+d).set(thing_classes=['balloon'])

cfg = get_cfg()   # 加载默认的config 
add_centernet_config(cfg)   # 添加centernet的config
cfg.merge_from_file("./configs/CenterNet2_R50_1x.yaml")  # 从config_file里合并一部分参数进来
# cfg.merge_from_file(model_zoo.get_config_file("./configs/CenterNet2_R50_1x.yaml")) #预设档，参数
cfg.MODEL.ROI_HEADS.NUM_CLASSES=5  #一类
# cfg.MODEL.DEVICE='cpu'  #注释掉此项，系统默认使用NVidia的显卡
cfg.MODEL.WEIGHTS='./output-balloon/CenterNet2/CenterNet2_R50_1x/model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7
predictor=DefaultPredictor(cfg)
val_dicts=DatasetCatalog.get('balloon_val')
balloon_metadata=MetadataCatalog.get('balloon_val')

s1,s2=0,0
for d in val_dicts:
    im=cv2.imread(d['file_name'])
    outputs=predictor(im)
    s1+=len(outputs['instances'].get("pred_classes"))
with open('./datasets/balloon/val/via_region_data.json') as f:
    im_js=json.load(f)
for i in im_js.keys():
    s2+=len(im_js[i]['regions'])
print(s1/s2)

for d in random.sample(val_dicts,3):
    im=cv2.imread(d['file_name'])
    outputs=predictor(im)
    v=Visualizer(im[:,:,::-1],metadata=balloon_metadata,scale=0.8)
    v=v.draw_instance_predictions(outputs['instances'].to('cpu'))
    plt.figure(figsize=(20,10))
    plt.imshow(v.get_image()[::,::-1])
    plt.show()
