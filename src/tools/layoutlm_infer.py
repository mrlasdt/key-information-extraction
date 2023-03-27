import os
import glob
from sklearn.feature_extraction import img_to_graph
from sklearn.metrics import confusion_matrix
import ssl
import random
import time
import string
from sympy import continued_fraction
import tqdm
import base64
import json
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import cv2
import torch
import shutil
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from transformers import (
    LayoutLMv2ForSequenceClassification, LayoutXLMTokenizer,
    LayoutLMv2FeatureExtractor, LayoutLMv2Processor, LayoutXLMProcessor,
    LayoutLMv2ForTokenClassification
)
from transformers import AdamW
import sys
sys.path.append('.')

from src.tools.utils import write_to_json_, construct_file_path
from create_kie_labels import create_kie_dict

sys.path.append('/home/sds/hoangmd/TokenClassification_copy')
# from src.experiments.word_formation import *
# import labelme
# from global_variables import *
# from process_label import *
from externals.tools.word_formation import *
import urllib
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""


from config import config as cfg
KIE_LABELS = cfg.KIE_LABELS
DEVICE = cfg.DEVICE
MAX_SEQ_LENGTH = 512  # TODO Fix this hard code


# DATA_ROOT = "data/207/idcard_cmnd_8-9-2022"
# # DATA_ROOT = "/mnt/hdd2T/AICR/TD_Project_PV/Verification_Dataset/PV2_Final/KIE_Identity_Card/Dataset/Processed/Vin/newForm"
# OCR_ROOT = "data/pseudo_label/207_extend"
# # OCR_ROOT = "data/pseudo_label/Vin/new_form"
# OCR_ROOT_TRAINED = "data/label/207/kie"
# # val_list = "val_list1.txt"
# LABEL2IDX = {KIE_LABELS[i]: i for i in range(len(KIE_LABELS))}
# # LABELME_VERSION = "4.6.0"
# # LABELME_FLAGS = ""

# SAVE_DIR = 'result/207_extend'
# SAVE_DIR = 'result/Vin/new_form'

# DATA_ROOT = 'data/GPLX/full'
# DATA_ROOT = 'data/v?al_df_300.csv'
# OCR_ROOT = "data/pse?udo_label/CCCD_cls_31_10_best_cpu"

DATA_ROOT = 'data/GPLX/full'
OCR_ROOT = "data/pseudo_label/GPLX_new_cls_magic"
# OCR_ROOT = "data/label/GPLX/kie/val"
SAVE_DIR = 'result/CCCD_cls_31_10_best_cpu'
# SAVE_DIR = 'result/CCCD_cls_31_10_best_cpu'

DF_PATH = "/mnt/hdd2T/datnt/KeyValueUnderstanding/IDCards_Missing_OCR/out.csv"
SAVE_DIR = '/mnt/hdd2T/datnt/KeyValueUnderstanding/IDCards_Missing_KIE'


def load_image_paths_and_labels_from_folder(data_root, ocr_root):
    r"""Load (image path, label) pairs into a DataFrame with keys ``image_path`` and ``label``

    @todo   Add OCR paths here
    """
    # f = open(val_list, "r+", encoding="utf-8")
    # lines = f.readlines()
    # lines = [line[:-1] for line in lines]
    # image_paths = [os.path.join(data_root,line) for line in lines]
    image_paths = glob.glob(os.path.join(data_root, "*.jpg"))
    ocr_paths = [
        os.path.join(ocr_root, os.path.basename(path)[:-3] + "txt")
        for path in image_paths
    ]
    return image_paths, ocr_paths


def load_image_paths_and_labels_from_df(df_path, ocr_root=None):
    df = pd.read_csv(df_path, index_col=0)
    image_paths = list(df.img_path.values)
    ocr_paths = list(df.ocr_path.values)
    if ocr_root:
        ocr_paths = [os.path.join(ocr_root, os.path.basename(path)) for path in ocr_paths]
    return image_paths, ocr_paths


def load_image_paths_and_labels(data_root: string, ocr_root: string):
    if data_root.endswith('.csv'):
        return load_image_paths_and_labels_from_df(data_root, ocr_root)
    elif os.path.isdir(data_root):
        return load_image_paths_and_labels_from_folder(data_root, ocr_root)
    else:
        raise NotImplementedError('Invalida DATA_ROOT')
# class Box():
#     def __init__(self, xmin=0, ymin=0, xmax=0, ymax=0, label="",kie_label=""):
#         self.xmin =xmin
#         self.ymin = ymin
#         self.xmax = xmax
#         self.ymax = ymax
#         self.label = label
#         self.kie_label = kie_label


def check_iou_1(box1: Box, box2: Box, threshold=0.8):
    area1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
    xmin_intersect = max(box1.xmin, box2.xmin)
    ymin_intersect = max(box1.ymin, box2.ymin)
    xmax_intersect = min(box1.xmax, box2.xmax)
    ymax_intersect = min(box1.ymax, box2.ymax)
    if xmax_intersect < xmin_intersect or ymax_intersect < ymin_intersect:
        area_intersect = 0
    else:
        area_intersect = (xmax_intersect - xmin_intersect) * (ymax_intersect * ymin_intersect)
    union = area1 + area2 - area_intersect
    iou = area_intersect / min(area1, area2)
    if iou > threshold:
        return True
    return False


def load_ocr_labels(ocr_path):
    r"""Load OCR labels, i.e. (word, bbox) pairs, into the input DataFrame containing of columns (image_path, ocr_path, label)"""
    with open(ocr_path) as f:
        lines = [
            line.replace('\n', '').replace('\r', '')
            for line in f.readlines()
        ]
    words, boxes, labels = [], [], []
    for i, line in enumerate(lines):
        line = line.split("\t")
        if len(line) == 6:  # TODO: make it new function in utils.py so that it is resuable
            x1, y1, x2, y2, text, _ = line
        elif len(line) == 5:
            x1, y1, x2, y2, text = line
        else:
            raise ValueError(f'Invalid ocr label format in {ocr_path}')
        label = "seller_name_value"  # TODO ??? fix this
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        if text != " ":
            words.append(text)
            boxes.append([x1, y1, x2, y2])
            # boxes.append(_normalize_box((x1, y1, x2, y2), w, h))
            labels.append(label)
    return words, boxes, labels


def _normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def _unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def infer(image_path, ocr_path, max_n_words=150):
    # Load inputs
    image = Image.open(image_path)
    # ori_image = image.copy()
    image = image.convert("RGB")
    batch_words, batch_boxes, batch_labels = load_ocr_labels(ocr_path)
    batch_preds, batch_true_boxes = [], []
    list_words = []
    for i in range(0, len(batch_words), max_n_words):
        words = batch_words[i:i + max_n_words]
        boxes = batch_boxes[i:i + max_n_words]
        boxes_norm = [_normalize_box(bbox, image.size[0], image.size[1]) for bbox in boxes]
        # boxes_norm =
        # labels = batch_labels[i:i + max_n_words]

        # Preprocess
        dummy_word_labels = [0] * len(words)
        start = time.time()
        encoding = processor(
            image, text=words, boxes=boxes_norm, word_labels=dummy_word_labels,
            return_tensors="pt", padding="max_length", truncation=True,
            max_length=512
        )

        # Run model

        for k, v in encoding.items():
            encoding[k] = v.to(DEVICE)
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        # token_boxes = encoding.bbox.squeeze().tolist()
        end1 = time.time() - start
        # Postprocess
        is_subword = (encoding['labels'] == -100).detach().cpu().numpy()[0]  # remove padding
        true_predictions = [
            pred for idx, pred in enumerate(predictions)
            if not is_subword[idx]
        ]
        # true_boxes = [
        #     _unnormalize_box(box, image.size[0], image.size[1])
        #     for idx, box in enumerate(token_boxes) if not is_subword[idx]
        # ]
        true_boxes = boxes  # TODO check assumption that layourlm do not change box order
        # we are doing this because if use unnormalized box, box that is too small will suffer rounding value problem -> result in area=0 box
        for i, word in enumerate(words):
            bndbox = [int(j) for j in true_boxes[i]]
            list_words.append(Word(text=word, bndbox=bndbox, kie_label=KIE_LABELS[true_predictions[i]]))
        end2 = time.time() - start
        batch_preds.extend(true_predictions)
        batch_true_boxes.extend(true_boxes)

    batch_preds = np.array(batch_preds)
    batch_true_boxes = np.array(batch_true_boxes)
    return batch_words, batch_preds, batch_true_boxes, list_words, end1, end2


def save_img_and_text(image_path, batch_words, batch_preds, batch_true_boxes, save_dir):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    txt_file = construct_file_path(save_dir, image_path, '.txt')
    f = open(txt_file, "w+", encoding="utf-8")
    for i, pred in enumerate(batch_preds):
        predict_label = KIE_LABELS[pred]
        word = batch_words[i]
        # print(batch_true_boxes[i])
        box = batch_true_boxes[i].tolist()
        f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(box[0], box[1], box[2], box[3], word, predict_label))
        if predict_label != cfg.IGNORE_KIE_LABEL:
            draw.rectangle(box, fill=None, outline='red')
            draw.text((int(box[0]) + 10, int(box[1]) - 10), fill='red', text=KIE_LABELS[pred], font=font)
    image.save(construct_file_path(save_dir, image_path))
    # ori_image.save(os.path.join("anh_result",os.path.basename(image_path)))
    f.close()


if __name__ == "__main__":
    if os.path.isdir(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
        os.mkdir(SAVE_DIR)
        print("[INFO]: Resetting folder...", SAVE_DIR)
    else:
        os.mkdir(SAVE_DIR)
        print("[INFO]: Creating folder...", SAVE_DIR)

    print(1)
    tokenizer = LayoutXLMTokenizer.from_pretrained(
        "weights/pretrained/layoutxlm-base/tokenizer", model_max_length=MAX_SEQ_LENGTH
    )
    print(2)
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    print(3)
    processor = LayoutXLMProcessor(feature_extractor, tokenizer)
    print(4)
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        cfg.KIE_WEIGHTS, num_labels=len(KIE_LABELS), local_files_only=True
    ).to(DEVICE)  # TODO FIX this hard code
    print(5)
    # image_paths, ocr_paths = load_image_paths_and_labels(DATA_ROOT, OCR_ROOT)
    image_paths, ocr_paths = load_image_paths_and_labels_from_df(DF_PATH)
    lend1, lend2 = [], []
    for image_path, ocr_path in tqdm.tqdm(zip(image_paths, ocr_paths)):
        if not os.path.exists(ocr_path):
            continue  # TODO REMOVE THIS
        print(ocr_path)
        batch_words, batch_preds, batch_true_boxes, list_words, end1, end2 = infer(image_path, ocr_path)
        lend1.append(end1)
        lend2.append(end2)
        list_words = throw_overlapping_words(list_words)
        save_img_and_text(image_path, batch_words, batch_preds, batch_true_boxes, save_dir=SAVE_DIR)
        kie_dict, bbox_dict = create_kie_dict(list_words)
        write_to_json_(construct_file_path(SAVE_DIR, image_path, '.json'), kie_dict)
        write_to_json_(construct_file_path(SAVE_DIR, image_path, '_bbox.json'), bbox_dict)
        # list_word_group = []
        # for line in list_lines:
        #     # print(line.text)
        #     for word_group in line.list_word_groups:
        #         word_group.update_kie_label()
        #         list_word_group.append(word_group)

        # kie_dict = giaykhasinh_construct_word_groups_to_kie_label(list_lines)

        # f = open(os.path.join(TEST_INFER_PATH,
        #          os.path.basename(image_path)[:-3] + "txt"), "w+", encoding="utf-8")
        # # f.write(os.path.basename(image_path)+"------------------------------"+str(index)+"\n")
        # for k, v in kie_dict.items():
        #     f.write("{}\t".format(k))
        #     # print(k+"   ")
        #     f.write("{}\t".format(v))
        #     f.write("\n")
        # index += 1
        # f.close()
    print(sum(lend1) / len(lend1))
    print(sum(lend2) / len(lend2))
