from detectron2.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from detectron2.evaluation.pascal_voc_evaluation import voc_ap
import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    objects = []
    obj_struct = {}
    for obj in tree.findall("object"):
        if obj.find("name").text == "person":
            for part in obj.findall("part"):
                obj_struct["name"]=part.find("name").text
                bbox = part.find("bndbox")
                obj_struct["bbox"] = [
                    int(bbox.find("xmin").text),
                    int(bbox.find("ymin").text),
                    int(bbox.find("xmax").text),
                    int(bbox.find("ymax").text),
                    ]    
                obj_struct["difficult"]=0
                objects.append(obj_struct)

    return objects

def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):

    with PathManager.open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = np.loadtxt(lines, dtype=np.str)

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename[0]] = parse_rec(annopath.format(imagename[0]))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        imagename=imagename[0]
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap

class MY_PascalVOCDetectionEvaluator(PascalVOCDetectionEvaluator):
    def __init__(self, dataset_name,Writer=None):
        super().__init__(dataset_name)
        self.Writer=Writer
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        # Too many tiny files, download all to local for speed.
        annotation_dir_local = PathManager.get_local_path(
            os.path.join(meta.dirname, "Annotations/")
        )
        self._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Layout", meta.split + ".txt")
        self._class_names = meta.thing_classes
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def MY_put_scalar(self,class_name,stats):
        self.Writer.put_scalar('AP IoU=0.50:0.95,class:{}'.format(class_name),stats[0])
        self.Writer.put_scalar('AP IoU=0.50,class:{}'.format(class_name),stats[1])
        self.Writer.put_scalar('AP IoU=0.75,class:{}'.format(class_name),stats[2])

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")
            class_AP={}
            stats=[]
            aps = defaultdict(list)  # iou -> ap per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)
                mAP = {iou: np.mean(x) for iou, x in aps.items()}
                class_AP[cls_name]={"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
               
                if self.Writer is not None:
                    stats[0]=np.mean(list(mAP.values()))
                    stats[1]=mAP[50]
                    stats[2]=mAP[75]
                    self.MY_put_scalar(cls_name,stats)
             
        
        print(class_AP)        
        ret = OrderedDict()

        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
        return ret


        