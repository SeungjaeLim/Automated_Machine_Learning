from google.colab import auth
auth.authenticate_user()

from google.colab import drive
drive.mount('/content/gdrive', force_remount=False)

import os
from pathlib import Path

folder = "colab/pytorch"
project_dir = "Auto_Sklearn"

base_path = Path("/content/gdrive/My Drive/")
project_path = base_path / folder / project_dir
os.chdir(project_path)
for x in list(project_path.glob("*")):
    if x.is_dir():
        dir_name = str(x.relative_to(project_path))
        os.rename(dir_name, dir_name.split(" ", 1)[0])
print(f"현재 디렉토리 위치: {os.getcwd()}")

import numpy as np
import pandas as pd

data = pd.read_csv('/content/gdrive/My Drive/colab/pytorch/Auto_Sklearn/Data/data.csv')

data.info()

data = data.drop(columns=['Unnamed: 32'])

data.diagnosis.value_counts(normalize=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[[col for col in data.columns if col not in ['id','diagnosis']]],
                                                    data['diagnosis'], test_size = 0.33)

import autosklearn.classification

help(autosklearn.classification.AutoSklearnClassifier)
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task = 300, tmp_folder = "./log/") # 5분 탐색을 하도록 해보자
automl.fit(x_train, y_train)

automl.leaderboard()

print(automl.sprint_statistics())

print(len(automl.get_models_with_weights()))
for info in automl.get_models_with_weights():
    print("\n")
    print("model, weight : {}\n{}".format(info[0], "-" * 10 + "\n" + str(info[1])))

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

model = automl

print(y_test.reset_index().groupby('diagnosis').size())
y_predict = model.predict(x_test)
y_predict_score = model.predict_proba(x_test)[:, 1]

print("Accuracy : {}, AUC : {}, Precision : {}, Recall : {}".format(
    accuracy_score(y_test, y_predict),
    roc_auc_score(y_test, y_predict_score),
    precision_score(y_test, y_predict, pos_label='B'),
    recall_score(y_test, y_predict, pos_label='B')
))


confusion_matrix(y_test, y_predict)

