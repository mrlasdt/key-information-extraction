import glob
import os
import nltk
import re
import statistics
import sys
sys.path.append('.')
from config import config as cfg

from utils import read_json, create_empty_kie_dict, construct_file_path, post_processing_date
# RESULT_JSON_PATH = 'result/Vin/new_form'
RESULT_JSON_PATH = 'result/GPLX_cls_31_10_epoch_9_cpu'
# RESULT_JSON_PATH = 'result/207_pseudo'
# RESULT_JSON_PATH = 'result/207_new_pseudo'
ED_LEVEL = 'word_ignore_symbol'
# ED_LEVEL = 'word'
# ED_LEVEL = 'char'
KIE_LABEL_LINE_PATH = 'data/label/GPLX/json_val_and_samsung_members'


LEXCEPTIONS_GT = [
    '187421917_1668044206721318_779901369147309116_n.json',
    '150094748_1728953773944481_6269983404281027305_n.json',
    '144003628_4199201026774245_6202264670231940239_n.json',
    '140687263_3755683421155059_7637736837539526203_n.json',
    '147585615_2757487791230682_5515346433540820516_n.json',
    '179642128_1945301335636182_5557211235870766646_n.json',
    '4378298-95583fbb1edb703a6c5bbc1744246058-1-1.json',
    # duplicate with 193926583_1626674417674644_309549447428666454_n.jpg
    '184606042_1586323798376373_2179113485447088932_n.json',
]


def post_processing(kie_label, str):
    str = str.replace('✪', ' ').strip()
    if kie_label == 'exp':
        if 'Không' in str:
            str = 'Không thời hạn'
        else:
            str_ = re.findall('(\d{2}\/\d{2}\/\d{4})', str)
            if len(str_) > 0:
                str = str_[0]
    elif kie_label == 'date':
        # str_ = re.findall('(\d{2,4})', str)
        # str = '/'.join(list(str_))
        return post_processing_date(str)
    elif kie_label == 'id':
        str = str.replace(' ', '')
    elif kie_label == 'add':
        str = str.replace('x.', 'X.')
        str = str.replace('x ', 'X. ')
        str = str.replace('X ', 'X. ')
        str = str.replace('P ', 'P. ')
        str = str.replace('p ', 'P. ')
        str = str.replace('p.', 'P.')
    return str


def tokenize(measure_unit, preds, truths):
    if measure_unit in ["word", "word_ignore_symbol"]:
        if measure_unit == "word_ignore_symbol":
            comp = re.compile('[^A-Z^a-z^0-9^\u00c0-\u1ef9 ]')
            preds = [comp.sub('', p) for p in preds]
            truths = [comp.sub('', t) for t in truths]
        unique_words = list(set(preds + truths))
        n_tokens = max(len(preds), len(truths))
        preds = [unique_words.index(w) for w in preds]
        truths = [unique_words.index(w) for w in truths]
    elif measure_unit == "char":
        # preds = " ".join(preds)
        # truths = " ".join(truths)
        # preds = preds.replace(" ", "")
        unique_chars = list(set(preds + truths))
        n_tokens = max(len(preds), len(truths))

        # print(n_tokens)
        preds = [unique_chars.index(w) for w in preds]
        truths = [unique_chars.index(w) for w in truths]
    elif measure_unit == "checkbox":
        unique_words = list(set(preds + truths))
        n_tokens = max(len(preds), len(truths))
        preds = [unique_words.index(w) for w in preds]
        truths = [unique_words.index(w) for w in truths]
    return n_tokens, preds, truths


def compute_metric(measure_unit, preds, truths):
    r"""Compute metric between AICREngine and human's labels

    Args:
        measure_unit (str): Unit of measurements. Possible values are "word", "char", or "checkbox"
        preds (list(str)): List of texts predicted by AICREngine
        truths (list(str)): List of texts labeled by human
    Returns:
        n_edits (int): Number of edits required to correct AICREngine's results
        n_tokens (int): Number of edittable units
        accuracy (float): (1 - n_edits / n_tokens)
    """
    assert measure_unit in ["word", "char", "checkbox", "word_ignore_symbol"], "Invalid measure unit"
    n_tokens, preds, truths = tokenize(measure_unit, preds, truths)
    n_edits = nltk.edit_distance(preds, truths)
    accuracy = 1 - n_edits / n_tokens
    return n_tokens, n_edits, accuracy


def eval_dir():
    error_dict = create_empty_kie_dict()
    number_dict = {kie_label: 0 for kie_label in error_dict}
    prs = glob.glob('{}/*.json'.format(RESULT_JSON_PATH))
    cnt = 0
    for pr in prs[::-1]:
        gt = os.path.join(KIE_LABEL_LINE_PATH, os.path.basename(pr))
        # if gt == '201417393_4052045311588809_501501345369021923_n.json':
        #     print('debug')
        if gt in LEXCEPTIONS_GT:
            continue
        if not os.path.exists(gt):
            continue
        prdict = read_json(pr)
        grdict = read_json(gt)
        error = []
        for k in prdict:
            if prdict[k] == "" and grdict[k] == "":
                continue
            prdict[k] = post_processing(k, prdict[k])
            grdict[k] = post_processing(k, grdict[k])
            if 'word' in ED_LEVEL:
                max_len_seq, n_edits, accuracy = compute_metric(ED_LEVEL, prdict[k].split(), grdict[k].split())
            elif 'char' in ED_LEVEL:
                max_len_seq, n_edits, accuracy = compute_metric('char', prdict[k], grdict[k])
            else:
                raise ValueError(f'Invalid ED_LEVEL: {ED_LEVEL}')
            # print(k, "   ", prdict[k], "  ", grdict[k], "   ", accuracy)
            if accuracy < 1.0 and k == "date":
                print('-' * 200)
                cnt += 1
                print(construct_file_path(RESULT_JSON_PATH, pr, '.jpg'))
                print(k, n_edits)
                print(pr)
                print(prdict[k])
                print(gt)
                print(grdict[k])
            error.append(accuracy)

            if len(error) != 0:
                mean_error = statistics.mean(error)
                error_dict[k].append(mean_error)
                number_dict[k] += 1
    new_number_dict = dict()
    new_error_dict = dict()
    for k, v in error_dict.items():
        if len(v) != 0:
            print(k, "\t", round(1 - statistics.mean(v), 2))  # , "\t", number_dict[k])
            new_number_dict[k] = number_dict[k]
            new_error_dict[k] = error_dict[k]
    print(cnt)
    # cal overall performance
    total = sum(list(new_number_dict.values()))
    overall = sum(new_number_dict[i] * statistics.mean(new_error_dict[i]) for i in list(new_error_dict.keys())) / total
    print("overall:", round(1 - overall, 2))


def eval_files():
    preds = read_json(
        "/home/sds/hoanglv/Projects/TokenClassification_invoice/runs/infer/kie_e2e_pred_17-10-2022-maxwords150_samplingv2_rm_dup_boxes/pred_e2e.json")
    gts = read_json("/home/sds/hoanglv/Projects/TokenClassification_invoice/DATA/test_end2end/test_e2e.json")
    KIE_LABELS = [
        # id invoice
        'no_key',
        'no_value',
        'form_key',
        'form_value',
        'serial_key',
        'serial_value',
        'date',

        # seller info
        'seller_company_name_key',
        'seller_company_name_value',
        # 'seller_name_key',
        # 'seller_name_value',
        'seller_tax_code_key',
        'seller_tax_code_value',
        'seller_address_value',
        'seller_address_key',
        'seller_mobile_key',
        'seller_mobile_value',
        # 'seller_company_name_value',   -> seller_name_value

        # buyer info
        'buyer_name_key',
        'buyer_name_value',
        'buyer_company_name_value',
        'buyer_company_name_key',
        'buyer_tax_code_key',
        'buyer_tax_code_value',
        'buyer_address_key',
        'buyer_address_value',
        'buyer_mobile_key',
        'buyer_mobile_value',

        # money info
        'VAT_amount_key',
        'VAT_amount_value',
        'total_key',
        'total_value',
        'total_in_words_key',
        'total_in_words_value',

        # 'other',
    ]
    error_dict = {k: [] for k in KIE_LABELS}
    number_dict = {kie_label: 0 for kie_label in error_dict}
    cnt = 0
    for file_name, gt in gts.items():
        error = []
        pred = preds[file_name]
        for k in gt:
            try:
                pred_text = pred[k]
            except KeyError:
                pred_text = ''
            if 'word' in ED_LEVEL:
                max_len_seq, n_edits, accuracy = compute_metric(ED_LEVEL, pred_text.split(), gt[k].split())
            elif 'char' in ED_LEVEL:
                max_len_seq, n_edits, accuracy = compute_metric('char', pred_text, gt[k])
            else:
                raise ValueError(f'Invalid ED_LEVEL: {ED_LEVEL}')
            error.append(accuracy)
            if len(error) != 0:
                mean_error = statistics.mean(error)
                error_dict[k].append(mean_error)
                number_dict[k] += 1
    new_number_dict = dict()
    new_error_dict = dict()
    for k, v in error_dict.items():
        if len(v) != 0:
            print(k, "\t", round(1 - statistics.mean(v), 2))  # , "\t", number_dict[k])
            new_number_dict[k] = number_dict[k]
            new_error_dict[k] = error_dict[k]
    total = sum(list(new_number_dict.values()))
    overall = sum(new_number_dict[i] * statistics.mean(new_error_dict[i]) for i in list(new_error_dict.keys())) / total
    print("overall:", round(1 - overall, 2))


if __name__ == "__main__":
    eval_dir()
    # eval_files()
# f = open("medical_doc/kie_pseudo_groundtruth.txt")
# lines = f.readlines()
# f.close()
# error_dict = {k:[] for k in list_value}
# number_dict = {k:0 for k in list_value}
# val_list = "val_list.txt"
# f = open(val_list,"r+",encoding = "utf-8")
# list_files = f.readlines()
# list_files = [file[:-1] for file in list_files]
# for i,line in enumerate(lines):
#     if '------' not in line:
#         continue


#     img_name = lines[i][:lines[i].index('-')]
#     file_name = img_name[:-3]
#     if img_name not in list_files:
#         continue

#     ## read groundtruth
#     groundtruth_dict=dict()
#     index = 1
#     while('-------' not in lines[i+index]):
#         l = lines[i+index].split("\t")
#         l = l[:-1] # remove "\n" character at the last position
#         if len(l)==1:
#             groundtruth_dict[l[0]]=[]
#         else:
#             groundtruth_dict[l[0]] = l[1:]
#         index+=1
#         if index+i == len(lines):
#             break


#     for k in list_value:
#         if k not in groundtruth_dict:
#             groundtruth_dict[k]=[]

#     ## read predict
#     f = open(os.path.join("medical_doc/kie_pseudo",file_name+"txt"))
#     pred_lines = f.readlines()
#     f.close()
#     predict_dict = dict()
#     for line in pred_lines:
#         l = line.split("\t")
#         l = l[:-1] # remove "\n" character at the last position
#         # print(l)
#         if len(l) ==1:
#             predict_dict[l[0]]=[]
#         else:
#             predict_dict[l[0]]=l[1:]
#     for k in list_value:
#         if k not in predict_dict:
#             predict_dict[k]=[]

#     print(file_name)
#     print ("groundtruth",groundtruth_dict['gen'])
#     print("predict", predict_dict['gen'])
#     #calculate error_ratio for each type of value
#     for k in list_value:
#         if len(predict_dict[k]) == 0 and len(groundtruth_dict[k])==0:
#             continue
#         groundtruth_len = len(groundtruth_dict[k])
#         predict_len = len(predict_dict[k])
#         error = []
#         max_len = max(groundtruth_len,predict_len)
#         predict_dict[k]+= [""]*(max_len-predict_len)
#         groundtruth_dict[k] += [""]*(max_len-groundtruth_len)

#         for str_groundtruth, str_predict in zip(groundtruth_dict[k],predict_dict[k]):
#             max_len_seq, n_edits, accuracy = count_edit(str_predict,str_groundtruth)
#             error.append(1-n_edits/max_len_seq)
#         if k =="buyer_company_name_value" or k=="seller_name_value":
#             print(k,"\t",groundtruth_dict[k],groundtruth_len,"\t",predict_dict[k],predict_len,"\t",error," max_seq:",max_len_seq)
#         mean_error = statistics.mean(error)
#         error_dict[k].append(mean_error)
#         number_dict[k]+=1


# new_number_dict =dict()
# new_error_dict=dict()
# for k, v in error_dict.items():
#     if len(v)!= 0:
#         print(k,"\t", statistics.mean(v), "\t", number_dict[k])
#         new_number_dict[k]= number_dict[k]
#         new_error_dict[k] = error_dict[k]

# total = sum(list(new_number_dict.values()))
# overall = sum(new_number_dict[i]*statistics.mean(new_error_dict[i]) for i in list(new_error_dict.keys())) / total
# print("overall:",overall)
