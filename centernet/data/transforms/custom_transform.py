# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Xingyi Zhou
# File: transform.py

import numpy as np
import torch
import torch.nn.functional as F
from fvcore.transforms.transform import (
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    TransformList,
)
from PIL import Image
import random
import math

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

__all__ = [
    "EfficientDetResizeCropTransform",
]


class EfficientDetResizeCropTransform(Transform):
    """
    """

    def __init__(self, scaled_h, scaled_w, offset_y, offset_x, img_scale, target_size, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        # assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4

        if img.dtype == np.uint8:
            pil_image = Image.fromarray(img)
            interp_method = interp if interp is not None else self.interp
            pil_image = pil_image.resize((self.scaled_w, self.scaled_h), interp_method)
            ret = np.asarray(pil_image)
            right = min(self.scaled_w, self.offset_x + self.target_size[1])
            lower = min(self.scaled_h, self.offset_y + self.target_size[0])
            # img = img.crop((self.offset_x, self.offset_y, right, lower))
            if len(ret.shape) <= 3:
                ret = ret[self.offset_y: lower, self.offset_x: right]
            else:
                ret = ret[..., self.offset_y: lower, self.offset_x: right, :]
        else:
            # PIL only supports uint8
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {Image.BILINEAR: "bilinear", Image.BICUBIC: "bicubic"}
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[self.interp]
            img = F.interpolate(img, (self.scaled_h, self.scaled_w), mode=mode, align_corners=False)
            shape[:2] = (self.scaled_h, self.scaled_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)
            right = min(self.scaled_w, self.offset_x + self.target_size[1])
            lower = min(self.scaled_h, self.offset_y + self.target_size[0])
            if len(ret.shape) <= 3:
                ret = ret[self.offset_y: lower, self.offset_x: right]
            else:
                ret = ret[..., self.offset_y: lower, self.offset_x: right, :]
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * self.img_scale
        coords[:, 1] = coords[:, 1] * self.img_scale
        coords[:, 0] -= self.offset_x
        coords[:, 1] -= self.offset_y
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        raise NotImplementedError
        # return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)

class RandomErasingTransform(Transform):
    def __init__(self, probability=0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean

    def apply_image(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    img[x1:x1+h, y1:y1+w,0] = random.uniform(0, 1)
                    img[x1:x1+h, y1:y1+w,1] = random.uniform(0, 1)
                    img[x1:x1+h, y1:y1+w,2] = random.uniform(0, 1)
                    # img[x1:x1+h, y1:y1+w,0] = self.mean[0]
                    # img[x1:x1+h, y1:y1+w,1] = self.mean[1]
                    # img[x1:x1+h, y1:y1+w,2] = self.mean[2]
                else:
                    img[x1:x1+h, y1:y1+w,0] = self.mean[1]
                return img

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_polygons(self, polygons):
        return polygons

class RandomNoiseTransform(Transform):
    def __init__(self, probability=0.5, noise_type='gaussian', mean=0, std=0.05):
        self.probability = probability
        self.noise_type = noise_type
        self.mean = mean
        self.std = std

    def apply_image(self, image):
        if random.random() < self.probability:
            if self.noise_type == 'gaussian':
                noise = np.random.normal(self.mean, self.std, size=image.shape)*255
                noise = noise.astype(np.uint8)
            elif self.noise_type == 'salt_and_pepper':
                noise = np.random.choice([0, 1], size=image.shape, p=[0.5, 0.5])
                noise = noise * 255
            else:
                noise = np.zeros_like(image)
            image = np.clip(image + noise, 0, 255)
        return image

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_polygons(self, polygons):
        return polygons