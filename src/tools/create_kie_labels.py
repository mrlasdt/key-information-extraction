# %%
# from pathlib import Path  # add Fiintrade path to import config, required to run main()
import sys
# FILE = Path(__file__).absolute()
# sys.path.append(FILE.parents[2].as_posix())  # add Fiintrade/ to path
sys.path.append('.')  # add Fiintrade/ to path


from src.tools.utils import load_kie_labels_yolo, create_empty_kie_dict, write_to_json_
import glob
import os
from externals.tools.word_formation import throw_overlapping_words, words_to_lines, Word

# DF_PATH = 'data/label/val_df_300.csv'
# KIE_LABEL_LINE_PATH = 'data/label/300/json'
KIE_LABEL_DIR = 'data/label/GPLX/kie/val'
KIE_LABEL_LINE_PATH = 'data/label/GPLX/json_val'

# %%

def merge_bbox(list_bbox):
    if not list_bbox:
        return list_bbox
    left = min(list_bbox, key=lambda x: x[0])[0]
    top = min(list_bbox, key=lambda x: x[1])[1]
    right = max(list_bbox, key=lambda x: x[2])[2]
    bot = max(list_bbox, key=lambda x: x[3])[3]
    return [left, top, right, bot]


def create_kie_dict(list_words):
    kie_dict = create_empty_kie_dict()
    # append each word to respected dict
    for word in list_words:
        if word.kie_label in kie_dict:
            kie_dict[word.kie_label].append(word)
        word.text = word.text.strip()

    # construct line from words for each kie_label
    bbox_dict = create_empty_kie_dict()
    for kie_label in kie_dict:
        list_lines, _ = words_to_lines(kie_dict[kie_label])
        kie_dict[kie_label] = '\n '.join([line.text.strip() for line in list_lines])
        bbox_dict[kie_label] = merge_bbox([line.boundingbox for line in list_lines])
    return kie_dict, bbox_dict

# %%


def main():
    # df = pd.read_csv(DF_PATH, index_col=0)
    # label_paths = list(df.label.values)
    label_paths = glob.glob(f'{KIE_LABEL_DIR}/*.txt')
    for label_path in label_paths:
        print(label_path)
        words, bboxes, kie_labels = load_kie_labels_yolo(label_path)
        list_words = []
        for i, kie_label in enumerate(kie_labels):
            list_words.append(Word(text=words[i], bndbox=bboxes[i], kie_label=kie_label))
        list_words = throw_overlapping_words(list_words)
        kie_dict = create_kie_dict(list_words)
        kie_path = os.path.join(KIE_LABEL_LINE_PATH, os.path.basename(label_path).replace('.txt', '.json'))
        write_to_json_(kie_path, kie_dict)

# %%


if __name__ == '__main__':
    main()
