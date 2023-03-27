export PYTHONPATH="$PYTHONPATH:/home/sds/thucpd/develop_OCR/TextDetectionApi/components/mmdetection"
export PYTHONPATH="$PYTHONPATH:/home/sds/datnt/mmocr"
# export PYTHONPATH="$PYTHONPATH:/home/sds/hoangmd/ocr"

export PYTHONWARNINGS="ignore"

#    --image  /mnt/hdd2T/AICR/TD_Project_PV/Verification_Dataset/PV2_Final/KIE_Identity_Card/Dataset/Processed/Vin/newForm \
#    --image  data/idcard_cmnd_8-9-2022 \


    # --image  data/GPLX/full \
    # --save_dir data/pseudo_label/GPLX_new_cls_magic \
python externals/tools/api.py \
    --image  data/val_df_300.csv \
    --save_dir data/pseudo_label/CCCD_cls_31_10_best_cpu \
    --det_cfg /home/sds/datnt/mmdetection/logs/textdet-add-synth-add-book-add-id/yolox_s_8x8_300e_cocotext_1280.py \
    --det_ckpt /home/sds/datnt/mmdetection/logs/textdet-add-synth-add-book-add-id/best_lite.pth \
    --cls_cfg /home/sds/datnt/mmocr/logs/satrn_big_2022-10-31/satrn_big.py \
    --cls_ckpt /home/sds/datnt/mmocr/logs/satrn_big_2022-10-31/best.pth
    # --cls_cfg /home/sds/datnt/mmocr/logs/satrn_big_2022-10-27/satrn_big.py \
    # --cls_ckpt /home/sds/datnt/mmocr/logs/satrn_big_2022-10-27/epoch_15.pth
    # --cls_cfg /home/sds/datnt/mmocr/logs/satrn_big_2022-04-25/satrn_big.py \
    # --cls_ckpt /home/sds/datnt/mmocr/logs/satrn_big_2022-04-25/best.pth 
    # --cls_cfg /home/sds/datnt/mmocr/logs/satrn_big_2022-10-13_kie/satrn_big.py \
    # --cls_ckpt /home/sds/datnt/mmocr/logs/satrn_big_2022-10-13_kie/best.pth 
    # --cls_cfg /home/sds/datnt/mmocr/logs/satrn_big_2022-10-04_kie/satrn_big.py \
    # --cls_ckpt /home/sds/datnt/mmocr/logs/satrn_big_2022-10-04_kie/best.pth