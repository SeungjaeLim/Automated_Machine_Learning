
from google.colab import auth
auth.authenticate_user()

from google.colab import drive
drive.mount('/content/gdrive', force_remount=False)

import os
from pathlib import Path

folder = "colab/pytorch"
project_dir = "Auto_Keras"

base_path = Path("/content/gdrive/My Drive/")
project_path = base_path / folder / project_dir
os.chdir(project_path)
for x in list(project_path.glob("*")):
    if x.is_dir():
        dir_name = str(x.relative_to(project_path))
        os.rename(dir_name, dir_name.split(" ", 1)[0])
print(f"현재 디렉토리 위치: {os.getcwd()}")

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re 
%matplotlib inline

import numpy as np
import pandas as pd

data = pd.read_csv('/content/gdrive/My Drive/colab/pytorch/Auto_Keras/Data/car_scrape.csv')

data.info()

data = data[data['year'] > 1900]

def make_label_map(dataframe):
    label_maps = {}
    for col in dataframe.columns:
        if dataframe[col].dtype=='object':
            label_map = {'unknown':0}
            for i, key in enumerate(dataframe[col].unique()):
                label_map[key] = i+1
            label_maps[col] = label_map
    
    return label_maps

def label_encoder(dataframe, label_map):
    for col in dataframe.columns:
        if dataframe[col].dtype=='object':
            dataframe[col] = dataframe[col].map(label_map[col])
            dataframe[col] = dataframe[col].fillna(label_map[col]['unknown'])
    
    return dataframe

le = make_label_map(data)
data = label_encoder(data, le)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[[col for col in data.columns if col not in ['price']]],
                                                    data['price'], test_size = 0.33)

from sklearn.svm import SVR

model = SVR()
model.fit(x_train, y_train)

def nmae(true, pred):

    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    
    return score

y_pred = model.predict(x_test)

svr_nmae = nmae(y_test, y_pred)


import autokeras as ak

reg = ak.StructuredDataRegressor(overwrite=True, max_trials=5)

reg.fit(x_train, y_train, 100)

autokeras_model = reg.export_model()

autokeras_model.summary()

tf.keras.backend.clear_session()

model = tf.keras.models.load_model('./structured_data_regressor/best_model/')

model.fit(x_train, y_train, epochs=100, validation_split=0.3)

autokeras_y_pred = model.predict(x_test)

autokeras_y_pred = autokeras_y_pred.squeeze()

autokeras_nmae = nmae(y_test, autokeras_y_pred)

print(f'SVR 모델 NMAE: {svr_nmae}')
print(f'AutoKeras NMAE: {autokeras_nmae}')