import re
from difflib import SequenceMatcher
from matplotlib.pyplot import cla

from rapidfuzz.distance import Levenshtein
import sys
sys.path.append('.')
from config import config as cfg
from terminaltables import AsciiTable
import json
import glob
import os
from utils import construct_file_path, write_to_txt_
gt_e2e_dir = "data/label/300/json"
# gt_e2e_dir = "data/label/GPLX/json_val_and_samsung_members"
pred_e2e_dir = "result/300_new_cls_magic"
# pred_e2e_dir = "result/GPLX_val_and_samsung_members"

# KIE_LABELS_WITH_ONLY_VALUES = [class_name for class_name in cfg.KIE_LABELS
#                                if class_name not in ['home', 'sex', 'eth', 'rel', 'org', 'others']]
KIE_LABELS_WITH_ONLY_VALUES = [class_name for class_name in cfg.KIE_LABELS
                               if class_name not in ['others']]
from edit_distance_to_Mai import LEXCEPTIONS_GT

import re
from difflib import SequenceMatcher
from matplotlib.pyplot import cla

from rapidfuzz.distance import Levenshtein


def post_processing_str(kie_label, str):
    str = str.replace('✪', ' ')
    if kie_label == 'exp':
        if 'Không' in str:
            str = 'Không thời hạn'
        else:
            str_ = re.findall('(\d{2}\/\d{2}\/\d{4})', str)
            if len(str_) > 0:
                str = str_[0]
    elif kie_label == 'date':
        str = str.replace(' /', ' ')
        pass
    return str


def post_process_dict(d):
    for k, v in d.items():
        d[k] = post_processing_str(k, v)
    return d


def post_processing_ddict(dd):
    for k, d in dd.items():
        dd[k] = post_process_dict(d)
    return dd


def read_json(file):
    with open(file, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data


def load_json_dir(dir_):
    res = {}
    for f in glob.glob(f"{dir_}/*.json"):
        d = read_json(f)
        res[f] = d
    return res


def is_type_list(x, type):

    if not isinstance(x, list):
        return False

    return all(isinstance(item, type) for item in x)


def cal_true_positive_char(pred, gt):
    """Calculate correct character number in prediction.
    Args:
        pred (str): Prediction text.
        gt (str): Ground truth text.
    Returns:
        true_positive_char_num (int): The true positive number.
    """

    all_opt = SequenceMatcher(None, pred, gt)
    true_positive_char_num = 0
    for opt, _, _, s2, e2 in all_opt.get_opcodes():
        if opt == 'equal':
            true_positive_char_num += (e2 - s2)
        else:
            pass
    return true_positive_char_num


def post_processing(text):
    '''
    - Remove special characters and  extra spaces + lower case
    '''

    text = re.sub(r"[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789 ]", " ", text)
    text = re.sub(r"\s\s+", " ", text)
    text = text.strip()

    return text


def count_matches(pred_texts, gt_texts, use_ignore=False):
    """Count the various match number for metric calculation.
    Args:
        pred_texts (list[str]): Predicted text string.
        gt_texts (list[str]): Ground truth text string.
    Returns:
        match_res: (dict[str: int]): Match number used for
            metric calculation.
    """
    match_res = {
        'gt_char_num': 0,
        'pred_char_num': 0,
        'true_positive_char_num': 0,
        'gt_word_num': 0,
        'match_word_num': 0,
        'match_word_ignore_case': 0,
        'match_word_ignore_case_symbol': 0,
        'match_kie': 0,
        'match_kie_ignore_case': 0

    }
    # comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
    # comp = re.compile('[]')
    norm_ed_sum = 0.0

    gt_texts_for_ned_word = []
    pred_texts_for_ned_word = []
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        if gt_text == pred_text:
            match_res['match_word_num'] += 1
            match_res['match_kie'] += 1
        gt_text_lower = gt_text.lower()
        pred_text_lower = pred_text.lower()
        if gt_text_lower == pred_text_lower:
            match_res['match_word_ignore_case'] += 1

        # gt_text_lower_ignore = comp.sub('', gt_text_lower)
        # pred_text_lower_ignore = comp.sub('', pred_text_lower)
        if use_ignore:
            gt_text_lower_ignore = post_processing(gt_text_lower)
            pred_text_lower_ignore = post_processing(pred_text_lower)

        else:
            gt_text_lower_ignore = gt_text_lower
            pred_text_lower_ignore = pred_text_lower

        if gt_text_lower_ignore == pred_text_lower_ignore:
            match_res['match_kie_ignore_case'] += 1

        gt_texts_for_ned_word.append(gt_text_lower_ignore.split(" "))
        pred_texts_for_ned_word.append(pred_text_lower_ignore.split(" "))

        match_res['gt_word_num'] += 1

        norm_ed = Levenshtein.normalized_distance(pred_text_lower_ignore,
                                                  gt_text_lower_ignore)
        # if norm_ed > 0.1:
        #     print(gt_text_lower_ignore, pred_text_lower_ignore, sep='\n')
        #     print("-"*20)
        norm_ed_sum += norm_ed

        # number to calculate char level recall & precision
        match_res['gt_char_num'] += len(gt_text_lower_ignore)
        match_res['pred_char_num'] += len(pred_text_lower_ignore)
        true_positive_char_num = cal_true_positive_char(
            pred_text_lower_ignore, gt_text_lower_ignore)
        match_res['true_positive_char_num'] += true_positive_char_num

    normalized_edit_distance = norm_ed_sum / max(1, len(gt_texts))
    match_res['ned'] = normalized_edit_distance

    # NED for word-level
    norm_ed_word_sum = 0.0
    # print(pred_texts_for_ned_word[0])
    unique_words = list(
        set([x for line in pred_texts_for_ned_word for x in line] + [x for line in gt_texts_for_ned_word for x in line]))
    preds = [[unique_words.index(w) for w in pred_text_for_ned_word]
             for pred_text_for_ned_word in pred_texts_for_ned_word]
    truths = [[unique_words.index(w) for w in gt_text_for_ned_word] for gt_text_for_ned_word in gt_texts_for_ned_word]
    for pred_text, gt_text in zip(preds, truths):
        norm_ed_word = Levenshtein.normalized_distance(pred_text,
                                                       gt_text)
        if norm_ed_word < 0.2:
            print(pred_text, gt_text)
        norm_ed_word_sum += norm_ed_word

    normalized_edit_distance_word = norm_ed_word_sum / max(1, len(gt_texts))
    match_res['ned_word'] = normalized_edit_distance_word

    return match_res


def eval_ocr_metric(pred_texts, gt_texts, metric='acc'):
    """Evaluate the text recognition performance with metric: word accuracy and
    1-N.E.D. See https://rrc.cvc.uab.es/?ch=14&com=tasks for details.
    Args:
        pred_texts (list[str]): Text strings of prediction.
        gt_texts (list[str]): Text strings of ground truth.
        metric (str | list[str]): Metric(s) to be evaluated. Options are:
            - 'word_acc': Accuracy at word level.
            - 'word_acc_ignore_case': Accuracy at word level, ignoring letter
              case.
            - 'word_acc_ignore_case_symbol': Accuracy at word level, ignoring
              letter case and symbol. (Default metric for academic evaluation)
            - 'char_recall': Recall at character level, ignoring
              letter case and symbol.
            - 'char_precision': Precision at character level, ignoring
              letter case and symbol.
            - 'one_minus_ned': 1 - normalized_edit_distance
            In particular, if ``metric == 'acc'``, results on all metrics above
            will be reported.
    Returns:
        dict{str: float}: Result dict for text recognition, keys could be some
        of the following: ['word_acc', 'word_acc_ignore_case',
        'word_acc_ignore_case_symbol', 'char_recall', 'char_precision',
        '1-N.E.D'].
    """
    assert isinstance(pred_texts, list)
    assert isinstance(gt_texts, list)
    assert len(pred_texts) == len(gt_texts)

    assert isinstance(metric, str) or is_type_list(metric, str)
    if metric == 'acc' or metric == ['acc']:
        metric = [
            'word_acc', 'word_acc_ignore_case', 'word_acc_ignore_case_symbol',
            'char_recall', 'char_precision', 'one_minus_ned'
        ]
    metric = set([metric]) if isinstance(metric, str) else set(metric)

    # supported_metrics = set([
    #     'word_acc', 'word_acc_ignore_case', 'word_acc_ignore_case_symbol',
    #     'char_recall', 'char_precision', 'one_minus_ned', 'one_minust_ned_word'
    # ])
    # assert metric.issubset(supported_metrics)

    match_res = count_matches(pred_texts, gt_texts)
    eps = 1e-8
    eval_res = {}

    if 'char_recall' in metric:
        char_recall = 1.0 * match_res['true_positive_char_num'] / (
            eps + match_res['gt_char_num'])
        eval_res['char_recall'] = char_recall

    if 'char_precision' in metric:
        char_precision = 1.0 * match_res['true_positive_char_num'] / (
            eps + match_res['pred_char_num'])
        eval_res['char_precision'] = char_precision

    if 'word_acc' in metric:
        word_acc = 1.0 * match_res['match_word_num'] / (
            eps + match_res['gt_word_num'])
        eval_res['word_acc'] = word_acc

    if 'word_acc_ignore_case' in metric:
        word_acc_ignore_case = 1.0 * match_res['match_word_ignore_case'] / (
            eps + match_res['gt_word_num'])
        eval_res['word_acc_ignore_case'] = word_acc_ignore_case

    if 'word_acc_ignore_case_symbol' in metric:
        word_acc_ignore_case_symbol = 1.0 * match_res[
            'match_word_ignore_case_symbol'] / (
                eps + match_res['gt_word_num'])
        eval_res['word_acc_ignore_case_symbol'] = word_acc_ignore_case_symbol

    if 'one_minus_ned' in metric:

        eval_res['1-N.E.D'] = 1.0 - match_res['ned']

    if 'one_minus_ned_word' in metric:

        eval_res['1-N.E.D_word'] = 1.0 - match_res['ned_word']

    if 'line_acc_ignore_case_symbol' in metric:
        line_acc_ignore_case_symbol = 1.0 * match_res[
            'match_kie_ignore_case'] / (
                eps + match_res['gt_word_num'])
        eval_res['line_acc_ignore_case_symbol'] = line_acc_ignore_case_symbol

    if 'line_acc' in metric:
        word_acc_ignore_case_symbol = 1.0 * match_res[
            'match_kie'] / (
                eps + match_res['gt_word_num'])
        eval_res['line_acc'] = word_acc_ignore_case_symbol

    for key, value in eval_res.items():
        eval_res[key] = float('{:.4f}'.format(value))

    return eval_res

# def eval(preds_e2e: dict, gt_e2e: dict):

#     pred_texts_dict = {
#         label: [] for label in KIE_LABELS_WITH_ONLY_VALUES
#     }
#     gt_texts_dict = {
#         label: [] for label in KIE_LABELS_WITH_ONLY_VALUES
#     }

#     results = {
#         label: 1 for label in KIE_LABELS_WITH_ONLY_VALUES
#     }
#     # print(KIE_LABELS_WITH_ONLY_VALUES)

#     for img_id in preds_e2e.keys():

#         pred_items = preds_e2e[img_id]
#         gt_items = gt_e2e[construct_file_path(gt_e2e_dir, img_id)]

#         for class_name, text_gt in gt_items.items():
#             if class_name not in KIE_LABELS_WITH_ONLY_VALUES:
#                 continue
#             if class_name not in pred_items:
#                 text_pred = ""
#             else:
#                 text_pred = pred_items[class_name]

#             pred_texts_dict[class_name].append(text_pred)
#             gt_texts_dict[class_name].append(text_gt)

#     for class_name in KIE_LABELS_WITH_ONLY_VALUES:
#         pred_texts = pred_texts_dict[class_name]
#         gt_texts = gt_texts_dict[class_name]
#         result = eval_ocr_metric(pred_texts, gt_texts, metric='acc')
#         results[class_name] = {
#             '1-ned': result['1-N.E.D'],
#             'word_acc': result['word_acc'],
#             'word_acc_ignore_case': result['word_acc_ignore_case'],
#             'word_acc_ignore_case_symbol': result['word_acc_ignore_case_symbol'],
#             'samples': len(pred_texts)
#         }

#     results['avg_all'] = {
#         '1-ned': cal_avg('1-ned', results),
#         'word_acc': cal_avg('word_acc', results),
#         'word_acc_ignore_case': cal_avg('word_acc_ignore_case', results),
#         'word_acc_ignore_case_symbol': cal_avg('word_acc_ignore_case_symbol', results),
#         'samples': sum([results[class_name]['samples'] for class_name in KIE_LABELS_WITH_ONLY_VALUES])
#     }

#     table_data = [["class_name", "1-NED", "word_acc", "word_acc_ignore_case", "word_acc_ignore_case_symbol", "#samples"]]
#     for class_name in results.keys():
#         # if c < p.shape[0]:
#         table_data.append(
#             [class_name, results[class_name]['1-ned'],
#              results[class_name]['word_acc'],
#              results[class_name]['word_acc_ignore_case'],
#              results[class_name]['word_acc_ignore_case_symbol'],
#              results[class_name]['samples']])

#     table = AsciiTable(table_data)
#     print(table.table)
#     return results


def eval(preds_e2e, gt_e2e):

    pred_texts_dict = {
        label: [] for label in KIE_LABELS_WITH_ONLY_VALUES
    }
    gt_texts_dict = {
        label: [] for label in KIE_LABELS_WITH_ONLY_VALUES
    }

    results = {
        label: 1 for label in KIE_LABELS_WITH_ONLY_VALUES
    }

    for img_id in preds_e2e.keys():
        pred_items = preds_e2e[img_id]
        gt_items = gt_e2e[construct_file_path(gt_e2e_dir, img_id)]

        for class_name, text_gt in gt_items.items():
            # if class_name  == 'seller_name_value':
            # print(gt_items)
            if class_name not in KIE_LABELS_WITH_ONLY_VALUES:
                continue
            if class_name not in pred_items:
                text_pred = ""
            else:
                text_pred = pred_items[class_name]

            pred_texts_dict[class_name].append(text_pred)
            gt_texts_dict[class_name].append(text_gt)

    for class_name in KIE_LABELS_WITH_ONLY_VALUES:
        pred_texts = pred_texts_dict[class_name]
        gt_texts = gt_texts_dict[class_name]
        result = eval_ocr_metric(
            pred_texts, gt_texts,
            metric=['one_minus_ned', 'line_acc_ignore_case_symbol', 'line_acc', 'one_minus_ned_word'])
        results[class_name] = {
            '1-ned': result['1-N.E.D'],
            '1-ned-word': result['1-N.E.D_word'],
            'line_acc': result['line_acc'],
            'line_acc_ignore_case_symbol': result['line_acc_ignore_case_symbol'],
            'samples': len(pred_texts)
        }

    # avg reusults
    sum_1_ned = sum([results[class_name]['1-ned'] * results[class_name]['samples']
                     for class_name in KIE_LABELS_WITH_ONLY_VALUES])
    sum_1_ned_word = sum([results[class_name]['1-ned-word'] * results[class_name]['samples']
                          for class_name in KIE_LABELS_WITH_ONLY_VALUES])

    sum_line_acc = sum([results[class_name]['line_acc'] * results[class_name]['samples']
                        for class_name in KIE_LABELS_WITH_ONLY_VALUES])
    sum_line_acc_ignore_case_symbol = sum(
        [results[class_name]['line_acc_ignore_case_symbol'] * results[class_name]['samples']
         for class_name in KIE_LABELS_WITH_ONLY_VALUES])

    total_samples = sum([results[class_name]['samples'] for class_name in KIE_LABELS_WITH_ONLY_VALUES])
    results['avg_all'] = {

        "1-ned": round(sum_1_ned / total_samples, 4),
        "1-ned-word": round(sum_1_ned_word / total_samples, 4),
        'line_acc': round(sum_line_acc / total_samples, 4),
        'line_acc_ignore_case_symbol': round(sum_line_acc_ignore_case_symbol / total_samples, 4),
        'samples': total_samples
    }

    table_data = [["class_name", "1-NED", '1-N.E.D_word', 'line-acc', 'line_acc_ignore_case_symbol', "#samples"]]
    for class_name in results.keys():
        # if c < p.shape[0]:
        table_data.append(
            [
                class_name,
                results[class_name]['1-ned'],
                results[class_name]['1-ned-word'],
                results[class_name]['line_acc'],
                results[class_name]['line_acc_ignore_case_symbol'],
                results[class_name]['samples']
            ]
        )

    table = AsciiTable(table_data)
    print(table.table)
    return results


if __name__ == "__main__":
    pred_e2e = load_json_dir(pred_e2e_dir)
    gt_e2e = load_json_dir(gt_e2e_dir)
    pred_e2e_cleaned = {}
    gt_e2e_cleaned = {}
    lgt_e2e_save = []
    for k, v in pred_e2e.items():
        k_gt = construct_file_path(gt_e2e_dir, k)
        if os.path.basename(k_gt) in LEXCEPTIONS_GT:
            continue
        if k_gt in gt_e2e:
            pred_e2e_cleaned[k] = v
            gt_e2e_cleaned[k_gt] = gt_e2e[k_gt]
            lgt_e2e_save.append(os.path.basename(k_gt))

    pred_e2e_cleaned = post_processing_ddict(pred_e2e_cleaned)
    gt_e2e_cleaned = post_processing_ddict(gt_e2e_cleaned)

    res = eval(pred_e2e_cleaned, gt_e2e_cleaned)

    print(res)
    # write_to_txt_('lgt_e2e.txt', lgt_e2e_save)
    with open('lgt_e2e.txt', 'w') as fp:
        for item in lgt_e2e_save:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
