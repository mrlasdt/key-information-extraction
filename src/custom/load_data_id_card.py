
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
from sklearn.model_selection import train_test_split
import pandas as pd

VN_LIST_CHAR = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!#$%&()*+,-./:;<=>?@[\]^_`{|}~'
def load_train_val_id_cards(train_root, label_path):
    train_labels = glob.glob(os.path.join(label_path, "*.txt"))
    img_names = [os.path.basename(train_label).replace('.txt', '.jpg') for train_label in train_labels]
    train_paths = [os.path.join(train_root, img_name) for img_name in img_names]
    train_df = pd.DataFrame.from_dict({'image_path': train_paths, 'label': train_labels})
    train, test = train_test_split(train_df, test_size=0.2, random_state=cfg.SEED)
    return train, test


def perturbate_character(list_word, ratio=0.01):
    # Algorithm
    # Step 1: couting number of characters is list_word and sample the postion we want to perturbation
    # ( list_word = ["abc", "lkdhf", "lfhdlsa", "akdjhf"] =>> total_char = 21, pertubation_position = [15,13,7]), these are the positions of perturbating chars in the concatenating string of list_word "abclkdhflfhdlsaakdjhf"
    # start_pos = 0, ending_pos = 0
    # Step 2: with each word in list_word, calculate the start_position and ending_pos of word in the concatenating string
    # if ending pos > pertubation_position[-1] => conduct perturbation and plus 1 to perturbation index, else => continue
    # Loop 1: list_word[0], start_pos = 0 ,ending_pos= 3 <= pertubation_position[-1] =7 => continue
    # Loop 2:

    total_char = sum(len(i) for i in list_word)

    pertubation_positions = sorted(random.sample(range(total_char), int(ratio * total_char)))
    # print(pertubation_positions)
    pos = 0
    start_pos = 0
    j = 0
    for i, word in enumerate(list_word):
        if j == len(pertubation_positions):
            break
        start_pos = pos
        pos += len(word)
        # print(start_pos,pos)
        while(pos > pertubation_positions[j]):
            x = random.randint(0, 3)
            fixing_pos = pertubation_positions[j] - start_pos
            if (x == 0):  # append random char to the left
                word = word[:fixing_pos] + VN_LIST_CHAR[random.randint(0, len(VN_LIST_CHAR) - 1)] + word[fixing_pos:]

            if (x == 1):  # append random char to the right
                word = word[:fixing_pos + 1] + VN_LIST_CHAR[random.randint(
                    0, len(VN_LIST_CHAR) - 1)] + word[fixing_pos + 1:]

            if (x == 2):  # adjust to another random char at current position
                word = word[:fixing_pos] + VN_LIST_CHAR[random.randint(0,
                                                                       len(VN_LIST_CHAR) - 1)] + word[fixing_pos + 1:]

            if (x == 3 and len(word) > 1):  # delete char at current position

                word = word[:fixing_pos] + word[fixing_pos + 1:]

            j += 1
            # print(list_word[i], word)
            list_word[i] = word
            if j == len(pertubation_positions):
                break

    return list_word


def _normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def load_ocr_labels(example, ocr_root, kie_labels, perturbate=0.0):
    r"""Load OCR labels, i.e. (word, bbox) pairs, into the input DataFrame containing of columns (image_path, ocr_path, label)"""
    image_id = os.path.splitext(os.path.basename(example["image_path"]))[0]
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
        # if label_kie == "mnv" or label_kie == "fnv":
        #     label_kie = "nv"
        # if label_kie == "mnk" or label_kie == "fnk":
        #     label_kie = "nk"
        # if label_kie == "fdobk" or label_kie == "fdobv":
        #     label_kie = "dobk"
        # if label_kie == "mdobk" or label_kie == "mdobv":
        #     label_kie = "dobv"

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box = _normalize_box((x1, y1, x2, y2), w, h)
        if text != " ":
            words.append(text)
            boxes.append(box)
            word_labels.append(kie_labels.index(label_kie))
    if perturbate > 0:
        p_words = perturbate_character(words, perturbate)
        print(len(p_words), len(words))
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


def load_data(train_path, label_path, max_seq_len, batch_size, pretrained_processor, kie_labels, device):
    train_df, val_df = load_train_val_id_cards(train_path, label_path)
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

