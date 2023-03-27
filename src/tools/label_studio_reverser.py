import pandas as pd
import json
import ast
import os
from tqdm import tqdm
# df = pd.read_csv('project-3-at-2022-09-11-15-09-0aff6824.csv')
# DF_PATH = 'data/label/Vin_newform/Vin_newform_full.csv'
# SAVE_PATH = 'data/label/Vin_newform/kie'
# PREFIX = 'http://localhost:8089/'

DF_PATH = 'data/label/GPLX/GPLX_samsung_members_label.csv'
SAVE_PATH = 'data/label/GPLX/kie/samsung_members'
PREFIX = 'http://localhost:8089/'


def label_studio_reverse(img_w, img_h, xywh):
    x1 = int(xywh[0] * img_w / 100)
    y1 = int(xywh[1] * img_h / 100)
    x2 = int(xywh[2] * img_w / 100 + x1)
    y2 = int(xywh[3] * img_h / 100 + y1)
    return (x1, y1, x2, y2)


def write_to_file(img_name, bboxes_, texts_, llabels_):
    fname = img_name.replace('.jpg', '.txt')
    with open(f'{SAVE_PATH}/{fname}', 'w') as f:
        for text, bbox, label in zip(texts_, bboxes_, llabels_):
            f.write(f'{bbox[0]}\t{bbox[1]}\t{bbox[2]}\t{bbox[3]}\t{text}\t{label}\n')


if __name__ == "__main__":
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    df = pd.read_csv(DF_PATH)
    for _, row in tqdm(df.iterrows()):
        bboxes = json.loads(row['bbox'])
        texts = ast.literal_eval(row['transcription'])
        dlabels = json.loads(row['label'])
        img_name = row['ocr'].split(PREFIX)[-1]
        bboxes_xyxy = []
        llabels = []
        for bbox, dlabel in zip(bboxes, dlabels):
            img_w = bbox['original_width']
            img_h = bbox['original_height']
            xywh = (bbox['x'], bbox['y'], bbox['width'], bbox['height'])
            xyxy = label_studio_reverse(img_w, img_h, xywh)
            bboxes_xyxy.append(xyxy)
            llabels.append(dlabel['labels'][0])
        write_to_file(img_name, bboxes_xyxy, texts, llabels)
