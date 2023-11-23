if __name__ == '__main__':
    #import multiprocessing as mp
    #mp.freeze_support()
    from dask.distributed import Client
    client = Client()

from ast import literal_eval
import os
import re
import dask.dataframe as dd
from dask_ml.preprocessing import OneHotEncoder
from dask_ml.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


df = dd.read_parquet('./animedata').compute()

for col in df:
    if df[col].nunique() == 1:
        print(f'{col} is constant')
        df = df.drop(columns=[col])

y = df['score']
print(type(y))
print(type(df))
X = df.drop(columns=['score', 'synopsis']).to_numpy()

lr = LinearRegression()
lr.fit(X, y)
print(lr.get_params())
print(lr.score(X, y))