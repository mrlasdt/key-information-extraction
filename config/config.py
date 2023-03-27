# GLOBAL VARS
DEVICE = "cpu"
# DEVICE = "cpu"
# DEVICE = "cpu"  # for debugging https://stackoverflow.com/questions/51691563/cuda-runtime-error-59-device-side-assert-triggered
# DEVICE = "cpu"
# KIE_LABELS = ['gen', 'nk', 'nv', 'dobk', 'dobv', 'other']
IGNORE_KIE_LABEL = 'others'
KIE_LABELS = ['id', 'name', 'dob', 'home', 'add', 'sex', 'nat', 'exp', 'eth', 'rel', 'date', 'org', IGNORE_KIE_LABEL]
KIE_WEIGHTS = "weights/ID_CARD_145_train_300_val_0.02_char_0.06_word"
# TODO: current yield index error if pass to gplx['data]['kie_label] (maybe mismatch with somewhere else) => fix this so that kie_label in gplx can be made global
# KIE_LABELS = ['id', 'name', 'dob', 'home', 'add', 'sex', 'nat',
#               'exp', 'eth', 'rel', 'date', 'org', IGNORE_KIE_LABEL, 'rank']
# KIE_WEIGHTS = 'weights/driver_license'
SEED = 42

##########################################
BASE_CONFIG = {
    'global': {
        'device': DEVICE,
        'kie_labels': KIE_LABELS,
    },
    "data": {
        "custom": True,
        "path": "src/custom/load_data.py",
        "method": "load_data",
        "train_path": "/home/sds/hoangmd/TokenClassification_copy/giaykhaisinh/synthesis_for_train/",
        "val_path": "/home/sds/hoangmd/TokenClassification_copy/giaykhaisinh/SDV_Meddoc_BirthCert/",
        # "size": 320,
        "max_seq_len": 512,
        "batch_size": 8,
        # "workers": 10,
        'pretrained_processor': 'microsoft/layoutxlm-base',
        'kie_labels': KIE_LABELS,
        'device': DEVICE,
    },

    "model": {
        "custom": True,
        "path": "src/custom/load_model.py",
        "method": "load_model",
        "pretrained_model": 'microsoft/layoutxlm-base',
        'kie_labels': KIE_LABELS,
        'device': DEVICE,
    },

    "optimizer": {
        "custom": True,
        "path": "src/custom/load_optimizer.py",
        "method": "load_optimizer",
        "lr": 5e-6,
        "weight_decay": 0,  # default = 0
        "betas": (0.9, 0.999),  # beta1 in transformer, default = 0.9
    },

    "trainer": {
        "custom": True,
        "path": "src/custom/load_trainer.py",
        "method": "load_trainer",
        "kie_labels": KIE_LABELS,
        "save_dir": 'weights',
        "n_epoches": 100,
    },
}

V2 = BASE_CONFIG
# V2['data'] = {
#     "custom": True,
#     "pretrained_model": 'microsoft/layoutxlm-base',
#     'kie_labels': KIE_LABELS,
#     'device': DEVICE,
# }

V3 = BASE_CONFIG
# V3["data"] = {
#     "custom": True,
#     "path": "src/custom/load_data_v3.py",
#     "method": "load_data",
#     "train_path": "/home/sds/hoangmd/TokenClassification_copy/giaykhaisinh/synthesis_for_train/",
#     "val_path": "/home/sds/hoangmd/TokenClassification_copy/giaykhaisinh/SDV_Meddoc_BirthCert/",
#     # "size": 320,
#     "max_seq_len": 512,
#     "batch_size": 8,
#     # "workers": 10,
#     'pretrained_processor': "microsoft/layoutlmv3-base",
#     'kie_labels': KIE_LABELS,
#     'device': DEVICE,
# }
# V3['model'] = {;
#     "custom": False,
#     'name': 'layoutlm_v3',
#     "pretrained_model": 'microsoft/layoutlmv3-base',
#     'kie_labels': KIE_LABELS,
#     'device': DEVICE,
# }

ID_CARD = BASE_CONFIG
ID_CARD['data'] = {
    "custom": True,
    "path": "src/custom/load_data_id_card.py",
    "method": "load_data",
    "train_path": "data/207/idcard_cmnd_8-9-2022",
    "label_path": "data/207/label/",
    # "size": 320,
    "max_seq_len": 512,
    "batch_size": 8,
    # "workers": 10,
    'pretrained_processor': 'microsoft/layoutxlm-base',
    'kie_labels': KIE_LABELS,
    'device': DEVICE,
}


GPLX = BASE_CONFIG
GPLX['data'] = {
    "custom": True,
    "path": "srcc/custom/load_data_gplx.py",
    "method": "load_data",
    "train_path": "data/GPLX/train/crop_blx_10_10_2022",
    "val_path": "data/GPLX/val/crop_blx_5_10_2022",
    "train_label_path": "data/label/GPLX/kie/train",
    "val_label_path": "data/label/GPLX/kie/val",
    # "size": 320,
    "max_seq_len": 512,
    "batch_size": 8,
    # "workers": 10,
    'pretrained_processor': 'microsoft/layoutxlm-base',
    'kie_labels': KIE_LABELS,
    'device': DEVICE,
}
