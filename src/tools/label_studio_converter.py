# https://labelstud.io/blog/Improve-OCR-quality-with-Tesseract-and-Label-Studio.html
from uuid import uuid4
from pathlib import Path
from PIL import Image
import sys
sys.path.append('.')
from src.tools.utils import write_to_json_, construct_file_path
import os
from tqdm import tqdm
# from config import config as cfg

SAVE_FILE = 'data/pseudo_label/json/GPLX_samsung_members.json'
DATA_PATH = 'result/GPLX_samsung_members'
# PSEUDO_PATH = 'result/Vin/new_form'
PSEUDO_PATH = 'result/GPLX_samsung_members'


DEFAULT_SCORE = 1
DEFAULT_TYPE = 'textarea'
DEFAULT_LABEL = 'others'
DEFAULT_TEXT = 'TEXT'
IMG_EXT = '.jpg'
LABEL_EXT = '.txt'


def get_name(path):
    return path.split('/')[-1]


def get_pseudo_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.read().splitlines()
    preds = []
    for line in lines:
        preds.append(line.split('\t'))
    return preds


def create_image_url(img_path):
    img_name = get_name(img_path)
    return f'http://localhost:8089/{img_name}'


def xyxy2xywh(bbox):
    return [
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]) - float(bbox[0]),
        float(bbox[3]) - float(bbox[1]),
    ]


def convert_to_ls(image, preds):
    """
    :param image: PIL image object
    :param tesseract_output: the output from tesseract
    :param per_level: control the granularity of bboxes from tesseract
    :return: tasks.json ready to be imported into Label Studio with "Optical Character Recognition" template
    """
    image_width, image_height = image.size
    results = []
    for pred in preds:
        if len(pred) == 6:
            x1, y1, x2, y2, text, kie_label = pred
        # elif len(pred) == 5:
        #     x1, y1, x2, y2, text = pred
        #     kie_label = DEFAULT_LABEL
        elif len(pred[0].split()) == 5:
            _, x1, y1, x2, y2 = pred[0].split()
            kie_label = DEFAULT_LABEL
            text = DEFAULT_TEXT
        else:
            raise ValueError('Invalid input {}'.format(pred))

        x, y, w, h = xyxy2xywh((x1, y1, x2, y2))
        bbox = {
            'x': 100 * x / image_width,
            'y': 100 * y / image_height,
            'width': 100 * w / image_width,
            'height': 100 * h / image_height,
            'rotation': 0
        }
        region_id = str(uuid4())[:10]
        bbox_result = {
            'id': region_id, 'from_name': 'bbox', 'to_name': 'image', 'type': 'rectangle',
            'value': bbox}
        transcription_result = {
            'id': region_id, 'from_name': 'transcription', 'to_name': 'image', 'type': 'textarea',
            'value': dict(text=[text], **bbox), 'score': DEFAULT_SCORE}
        label_result = {
            'id': region_id, 'from_name': 'label', 'to_name': 'image', 'type': 'labels',
            'value': dict(labels=[kie_label], **bbox)}
        results.extend([bbox_result, transcription_result, label_result])

    return {
        'data': {
            'ocr': create_image_url(get_name(image.filename))
        },
        'predictions': [{
            'result': results,
            'score': DEFAULT_SCORE
        }]
    }


def main():
    tasks = []
    # collect the receipt images from the image directory
    for f in tqdm(Path(PSEUDO_PATH).glob(f'*{LABEL_EXT}')):
        preds = get_pseudo_label(f.absolute())
        image_path = construct_file_path(DATA_PATH, os.path.basename(f.absolute()), IMG_EXT)
        with Image.open(image_path) as image:
            task = convert_to_ls(image, preds)
            tasks.append(task)
    # create a file to import into Label Studio

    # with open(SAVE_FILE, mode='w', encoding='utf8') as f:
    #     json.dump(tasks, f, indent=2, ensure_ascii=False)
    write_to_json_(SAVE_FILE, tasks)


if __name__ == "__main__":
    main()
