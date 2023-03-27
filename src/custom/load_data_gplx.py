
import os
from datasets import Dataset
import random
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from functools import partial
from PIL import Image
import cv2
from transformers import LayoutXLMProcessor, LayoutXLMTokenizer, LayoutLMv2FeatureExtractor
import torch
import glob
import pandas as pd
from src.tools.utils import construct_file_path


def load_img_and_label_to_df_gplx(data_root, label_root):
    label_paths = glob.glob(os.path.join(label_root, "*.txt"))
    label_paths_with_existing_img = []
    data_paths = []
    for l in label_paths:
        img_path = construct_file_path(data_root, l, '.jpg')
        if os.path.exists(img_path):
            data_paths.append(img_path)
            label_paths_with_existing_img.append(l)
    df = pd.DataFrame.from_dict({'image_path': data_paths, 'label': label_paths_with_existing_img})
    return df


# def load_img_and_label_to_df_gplx(data_root, label_root):
#     label_paths = glob.glob(os.path.join(label_root, "*.txt"))
#     data_paths = [construct_file_path(data_root, l, '.jpg') for l in label_paths]
#     df = pd.DataFrame.from_dict({'image_path': data_paths, 'label': label_paths})
#     return df


def _normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def load_ocr_labels(example, ocr_root, kie_labels):
    r"""Load OCR labels, i.e. (word, bbox) pairs, into the input DataFrame containing of columns (image_path, ocr_path, label)"""
    ocr_path = os.path.join(ocr_root, f"{example['label']}")
    assert os.path.exists(ocr_path)

    image = cv2.imread(example["image_path"])
    h, w, _ = image.shape
    with open(ocr_path) as f:
        lines = [
            line.replace('\n', '').replace('\r', '')
            for line in f.readlines()
        ]
    words, boxes, word_labels = [], [], []
    for i, line in enumerate(lines):
        x1, y1, x2, y2, text, label_kie = line.split("\t")
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box = _normalize_box((x1, y1, x2, y2), w, h)
        if text != " ":
            words.append(text)
            boxes.append(box)
            word_labels.append(kie_labels.index(label_kie))
    example["words"] = words
    example["bbox"] = boxes  # TODO: Check this
    example["word_labels"] = word_labels
    return example


def preprocess_data(examples, max_seq_length, processor):
    # Add custom OCR here
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    words = examples['words']
    normalized_word_boxes = examples['bbox']
    word_labels = examples['word_labels']

    assert all([
        len(_words) == len(boxes)
        for _words, boxes in zip(words, normalized_word_boxes)
    ])
    assert all([
        len(_words) == len(_word_labels)
        for _words, _word_labels in zip(words, word_labels)
    ])

    # Process examples
    encoded_inputs = processor(
        images, padding="max_length", truncation=True, text=words,
        boxes=normalized_word_boxes, word_labels=word_labels,
        max_length=max_seq_length
    )
    # print("encode inputtttttt", encoded_inputs.keys())
    return encoded_inputs


def load_data(
        train_path, val_path, train_label_path, val_label_path, max_seq_len, batch_size, pretrained_processor,
        kie_labels, device):
    train_df = load_img_and_label_to_df_gplx(train_path, train_label_path)
    val_df = load_img_and_label_to_df_gplx(val_path, val_label_path)
    print('Train: ', len(train_df))
    print('Val: ', len(val_df))
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print("Loading OCR labels ...")
    train_dataset = train_dataset.map(lambda example: load_ocr_labels(example, "", kie_labels))
    val_dataset = val_dataset.map(lambda example: load_ocr_labels(example, "", kie_labels))

    print("Preparing dataset ...")
    features = Features({
        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(max_seq_len, 4)),
        'labels': Sequence(ClassLabel(names=kie_labels))
    })
    # processor = LayoutXLMProcessor.from_pretrained(pretrained_processor)
    tokenizer = LayoutXLMTokenizer.from_pretrained(pretrained_processor, model_max_length=max_seq_len)
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    processor = LayoutXLMProcessor(feature_extractor, tokenizer)
    preprocess_data_default = partial(preprocess_data, max_seq_length=max_seq_len, processor=processor)
    train_dataset = train_dataset.map(
        preprocess_data_default, remove_columns=train_dataset.column_names, features=features,
        batched=True, batch_size=batch_size
    )
    train_dataset.set_format(type="torch", device=device)

    val_dataset = val_dataset.map(
        preprocess_data_default, remove_columns=val_dataset.column_names, features=features,
        batched=True, batch_size=batch_size
    )
    val_dataset.set_format(type="torch", device=device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader
