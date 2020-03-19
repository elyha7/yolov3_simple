# Original repo: https://github.com/ultralytics/yolov3
from utils.utils import *
from utils import torch_utils
import torch
import cv2
from models.models import Darknet
import numpy as np
import os
import torchvision
import random
from PIL import Image
import json
import sys

class YoloV3Predictor:
    def __init__(self,
                 device=None,
                 target_size=416,
                 half_precision=False,
                 use_onnx=False,
                 classes=None,
                 cfg_path='cfg/yolov3-spp.cfg',
                 weights_path='weights/ultralytics68.pt'):
        """
            Object detector YoloV3 trained on COCO dataset.
            Args:
                device (int, str): device to run inference on. int num for gpu use,  "cpu" for cpu.
                target_size (int): target size of biggest side of the image e.g 416, 640, 720.
                half_precision (bool): use torch half precision engine
                use_onnx (bool): convert model to onnx.
                classes (list of int): class numbers to detect on image. see data/labels.json.
                cfg_path (str): path to yolo config.
                weights_path (str): path to pretrained model.
        """
        self.half = half_precision
        self.classes = classes
        if use_onnx:
            self.target_size = (416, 256)
            self.device = self.set_device('cpu')
        else:
            self.target_size = round(target_size/32)*32
            self.device = self.set_device(device)
        self.model = self.load_model(use_onnx, half_precision)

    def set_device(self, device):
        """
            Set torch.device to cpu/gpu mode.
        """
        if device is None:
            if torch.cuda.is_available():
                device = 0
            else:
                device = 'cpu'
        if device == 'cpu':
            torch_device = torch.device('cpu')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
            torch_device = torch.device('cuda:0')
        return torch_device

    def load_model(self, use_onnx, half_precision,cfg_path,weights_path):
        """
            Load model weights into device memory.
        """
        model = Darknet(cfg_path, self.target_size)
        model.load_state_dict(torch.load(
            weights_path, map_location=self.device)['model'])
        model.to(self.device).eval()
        if use_onnx:
            model.fuse()
            img = torch.zeros((1, 3) + self.target_size)  
            torch.onnx.export(model, img, 'weights/export.onnx',
                              verbose=False, opset_version=10)

            import onnx
            model = onnx.load('weights/export.onnx') 
            onnx.checker.check_model(model)  
        if half_precision and self.device.type != 'cpu':
            model.half()
        torch.backends.cudnn.benchmark = True
        return model

    def preprocess_image(self, img):
        """
            Resize and pad image to make it 32-multiple for speedup and
            scale image array values from [0,255] to [0,1].
        """
        img = letterbox(img, new_shape=self.target_size,
                        interp=cv2.INTER_LINEAR)[0]
        img = np.rollaxis(img, 2, 0)
        img = np.ascontiguousarray(
            img, dtype=np.float16 if self.half else np.float32)
        img /= 255.0
        return img

    def predict(self, img, iou_thres=0.5, conf_thres=0.3, agnostic_iou=True):
        """
            Predict bounding boxes and classes from image.
            Params:
                img (ndarray) : original image.
                iou_thresh (float) : Filters objects by intersecting bounding boxes area.
                conf_thresh (float) : Filters objects by classification confidence value.
                agnostic_iou (bool) : Pruning away boxes that have high IOU overlap with already selected boxes.  
                It operates on all the boxes using max scores across all classes for which scores are provided.
            Returns:
                pred (list of ndarrays) : each array is vector with length 6 and following structure: 
                (x1,y2,x2,y2,confidence,class).
        """
        orig_shape = img.shape
        img = self.preprocess_image(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.unsqueeze(0)
        pred = self.model(img)[0]
        if self.half:
            pred = pred.float()
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=self.classes, agnostic=agnostic_iou)
        for i, det in enumerate(pred):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], orig_shape).round()
        return pred

if __name__ == "__main__":
    model = YoloV3Predictor(0)