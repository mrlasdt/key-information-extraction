# import cv2
# img = cv2.imread(
#     '/home/sds/hoanglv/Projects/document_detection/idcard_inference/output/idcard_cmnd_8-9-2022/241159973_3002084596582610_3116559855862520677_n.jpg')

# with open('data/pseudo_label/241159973_3002084596582610_3116559855862520677_n.txt', 'r') as f:
#     label = f.read().splitlines()
# label = [line.split('\t') for line in label]

# for l in label:
#     img = cv2.rectangle(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), color=(255, 0, 0), thickness=1)
# cv2.imwrite('out.jpg', img)


# %%
import cv2
pred = {'id': {'text': '001096008006', 'bbox': [321, 112, 609, 143]},
        'name': {'text': 'VŨ MINH ANH', 'bbox': [412, 175, 570, 200]},
        'dob': {'text': '15/08/1996', 'bbox': [424, 221, 538, 243]},
        'home': {'text': 'Văn Giang, Hưng Yên', 'bbox': [353, 298, 582, 323]},
        'add': {'text': 'Tổ 8\n Thịnh Quang, Đống Đa, Hà Nội', 'bbox': [305, 349, 626, 409]},
        'sex': {'text': 'Nam', 'bbox': [311, 261, 360, 281]},
        'nat': {'text': 'Việt Nam', 'bbox': [514, 259, 610, 284]},
        'exp': {'text': '15/08/2021', 'bbox': [137, 404, 237, 419]},
        'eth': {'text': '', 'bbox': []},
        'rel': {'text': '', 'bbox': []},
        'date': {'text': '', 'bbox': []},
        'org': {'text': '', 'bbox': []}}
img_path = "data/207/idcard_cmnd_8-9-2022/130831493_220311142886087_6307056579275234360_n.jpg"
img = cv2.imread(img_path)
for kie_label, text_and_bbox in pred.items():
    l = text_and_bbox["bbox"]
    if l == []:
        continue
    img = cv2.rectangle(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), color=(255, 0, 0), thickness=1)
cv2.imwrite('out.jpg', img)
# %%
