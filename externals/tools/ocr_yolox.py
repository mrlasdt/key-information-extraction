# temp #for debug
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from mmocr.apis import init_detector as init_classifier
from mmocr.apis.inference import model_inference
import cv2
from PIL import Image
from io import BytesIO
from config import config as cfg
import numpy as np
from .utils import read_image_file, get_crop_img_and_bbox


class YoloX():
    def __init__(self, config, checkpoint):
        self.model = init_detector(config, checkpoint, cfg.DEVICE)

    def inference(self, img=None):
        return inference_detector(self.model, img)


class Classifier_SATRN:
    def __init__(self, config, checkpoint):
        self.model = init_classifier(config, checkpoint, cfg.DEVICE)

    def inference(self, numpy_image):
        result = model_inference(self.model, numpy_image, batch_mode=True)
        preds_str = [r["text"] for r in result]
        confidence = [r["score"] for r in result]
        return preds_str, confidence


class OcrEngineForYoloX():
    def __init__(self, det_cfg, det_ckpt, cls_cfg, cls_ckpt):
        self.det = YoloX(det_cfg, det_ckpt)
        self.cls = Classifier_SATRN(cls_cfg, cls_ckpt)

    def run_image(self, img_path):
        img = read_image_file(img_path)
        pred_det = self.det.inference(img)
        bboxes = np.vstack(pred_det)
        lbboxes = []
        lcropped_img = []
        assert len(bboxes) != 0, f'No bbox found in {img_path}, skipped'
        for bbox in bboxes:
            try:
                crop_img, bbox_ = get_crop_img_and_bbox(img, bbox, extend=True)
                lbboxes.append(bbox_)
                lcropped_img.append(crop_img)
            except AssertionError:
                print(f'[ERROR]: Skipping invalid bbox {bbox} in ', img_path)
        lwords, _ = self.cls.inference(lcropped_img)
        return lbboxes, lwords
