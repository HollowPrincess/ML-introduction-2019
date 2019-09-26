import numpy as np

def normalize(df):
    std=np.std(df).replace(0, 1)
    mean=np.mean(df)
    return (df-mean)/std, mean, std

def normalize_with_params(df, mean, std):
    return (df-mean)/std