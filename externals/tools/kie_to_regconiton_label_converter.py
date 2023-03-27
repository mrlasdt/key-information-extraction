# %%
import sys
sys.path.append('.')
from src.tools.utils import load_kie_labels_yolo, construct_file_path, write_to_txt_, get_name
import cv2
import glob
import os
import shutil
from externals.tools.utils import read_image_file, get_crop_img_and_bbox
import unidecode
# %%
KIE_LABEL_DIR = 'data/label/GPLX/kie/train_text_recog'
REG_LABEL_DIR = 'data/label/GPLX/recognition/train'
IMAGE_DATA_DIR = '/home/sds/hoanglv/Projects/document_detection/idcard_inference/output/crop_blx_10_10_2022_2'

# REG_LABEL_DIR = 'data/label/207/recognition'
# KIE_LABEL_DIR = 'data/label/207/kie'
# IMAGE_DATA_DIR = 'data/207/idcard_cmnd_8-9-2022'
# %%


def construct_reg_label_path(name, word_without_accent):
    file_name = name + '_' + word_without_accent
    return (os.path.join(REG_LABEL_DIR, file_name) + ext for ext in ['.txt', '.jpg'])


if __name__ == "__main__":
    if os.path.isdir(REG_LABEL_DIR):
        shutil.rmtree(REG_LABEL_DIR)
        os.mkdir(REG_LABEL_DIR)
    else:
        os.mkdir(REG_LABEL_DIR)
    kie_label_paths = glob.glob(f'{KIE_LABEL_DIR}/*.txt')
    for kie_label_path in kie_label_paths:
        words, boxes, labels = load_kie_labels_yolo(kie_label_path)
        img_path = construct_file_path(IMAGE_DATA_DIR, kie_label_path, '.jpg')
        img = read_image_file(img_path)
        print(len(boxes))
        for bbox, word, label in zip(boxes, words, labels):
            if label != 'others':
                crop_img, _ = get_crop_img_and_bbox(img, bbox, extend=False)
                word_without_accent = unidecode.unidecode(word).replace('/', '_').replace(':', '_')
                name = get_name(kie_label_path, ext=False)
                label_path, img_path = construct_reg_label_path(name, word_without_accent)
                post_fix = -1
                while os.path.exists(label_path):
                    post_fix += 1
                    word_without_accent_ = word_without_accent + str(post_fix)
                    label_path, img_path = construct_reg_label_path(name, word_without_accent_)
                cv2.imwrite(img_path, crop_img)
                write_to_txt_(label_path, word)
