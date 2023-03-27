from config import config as cfg
import json
import os



def load_kie_labels_yolo(label_path):
    with open(label_path, 'r') as f:
        lines = f.read().splitlines()
    words, boxes, labels = [], [], []
    for line in lines:
        x1, y1, x2, y2, text, kie = line.split("\t")
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if text != " ":
            words.append(text)
            boxes.append((x1, y1, x2, y2))
            labels.append(kie)
    return words, boxes, labels


def create_empty_kie_dict():
    return {cfg.KIE_LABELS[i]: [] for i in range(len(cfg.KIE_LABELS)) if cfg.KIE_LABELS[i] != cfg.IGNORE_KIE_LABEL}


def write_to_json_(file_path, content):
    with open(file_path, mode='w', encoding='utf8') as f:
        json.dump(content, f, ensure_ascii=False)


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def get_name(file_path, ext: bool = True):
    file_path_ = os.path.basename(file_path)
    return file_path_ if ext else os.path.splitext(file_path_)[0]


def construct_file_path(dir, file_path, ext=''):
    '''
    args:
        dir: /path/to/dir
        file_path /example_path/to/file.txt
        ext = '.json'
    return 
        /path/to/dir/file.json
    '''
    return os.path.join(
        dir, get_name(file_path,
                      True)) if ext == '' else os.path.join(
        dir, get_name(file_path,
                      False)) + ext


def write_to_txt_(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)


