import pandas as pd
import numpy as np

import pickle
import json

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

test_data=pd.read_csv("./data/processed/test_processed.csv")

x_test=test_data.iloc[:,0:-1].values
y_test=test_data.iloc[:,-1].values

model=pickle.load(open("model.pkl","rb"))
y_pred=model.predict(x_test)

accu=accuracy_score(y_test,y_pred)
prece=precision_score(y_test,y_pred)
rec_scr=recall_score(y_test,y_pred)
F1_Score=f1_score(y_test,y_pred)

metrics_dict = {

            'acc':accu,
            'precision':prece,
            'recall' : rec_scr,
            'f1_score': F1_Score
        }

with open('metric.json','w') as file:
        json.dump(metrics_dict,file,indent=4)