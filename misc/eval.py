# %%
from glob import glob
import requests
import glob
import json
from tqdm import tqdm
from pathlib import Path
import pandas as pd
# %%


def load_image_path():
    df = pd.read_csv('data/label/207_backup/207_val_df.csv', index_col=0)
    return list(df.image_path.values)


def login():
    try:
        login_url = 'http://107.120.70.69:8084/login'
        headers = {'Content-Type': 'application/json', 'accept': 'application/json'}
        data = {'username': 'admin', 'password': 'admin'}
        with requests.Session() as s:
            login_res = s.post(login_url, headers=headers, json=data)
        return login_res.json()['token']
    except Exception as e:
        print('[ERROR]: Cannot login\n', e)


def predict(bearer=None):
    image_paths = load_image_path()
    for image_path in tqdm(image_paths, ascii=True):
        name_image = Path(image_path).stem
        files = {'id_1': 'admin',
                 'id_2': 'admin',
                 'file': open(image_path, 'rb'), }
        headers = {'Content-Type': 'multipart/form-data', 'accept': 'application/json', }
        #    'Authorization': 'Bearer {}'.format(bearer)}
        with requests.Session() as s:
            response = s.post('http://localhost:8084/predict/image', headers=headers, files=files)
        with open('result/PV2/{}.json'.format(name_image), 'w') as f:
            json.dump(response._content.decode("utf-8"), f, indent=4)
        print(response, response.content)
        break


# %%
if __name__ == '__main__':
    predict()

# %%
