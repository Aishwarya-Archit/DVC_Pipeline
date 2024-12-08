import pandas as pd
import numpy as np
import os

train_data=pd.read_csv("./data/raw/train.csv")
test_data=pd.read_csv("./data/raw/test.csv")


def fill_missing_with_med(df):
    for column in df.columns:
        if df[column].isnull().any():
            med_value=df[column].median()
            df[column].fillna(med_value,inplace=True)
    return df

train_processed_data=fill_missing_with_med(train_data)
test_processed_data=fill_missing_with_med(test_data)

data_path_processed=os.path.join("data","processed")
os.makedirs(data_path_processed)

train_processed_data.to_csv(os.path.join(data_path_processed,"train_processed.csv"),index=False)
test_processed_data.to_csv(os.path.join(data_path_processed,"test_processed.csv"),index=False)