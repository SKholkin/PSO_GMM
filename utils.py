from sklearn.preprocessing import MinMaxScaler
import os.path as osp
import pandas as pd
import numpy as np

real_dataset_names = ['breast_cancer', 'cloud']

def load_breast_cancer():
    bc_data_name = 'data/wdbc.data'
    with open(bc_data_name) as f:
        data_bc = pd.DataFrame([item.split(',')[2:] for item in f.readlines()])
    data_bc = data_bc.astype(float).to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(data_bc)
    data_bc = scaler.transform(data_bc)
    return data_bc

def load_cloud_dataset():
    cloud_data_name = 'data/cloud.data'
    with open(cloud_data_name) as f:
        cloud_data = pd.DataFrame([item.split() for item in f.readlines()])
    cloud_data = cloud_data.astype(float).to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(cloud_data)
    cloud_data = scaler.transform(cloud_data)
    return cloud_data

def load_synthetic_dataset(filename):
    data = np.load(filename)
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data

def load_dataset(dataset_name):
    if dataset_name == 'breast_cancer':
        return load_breast_cancer()
    if dataset_name == 'cloud':
        return load_cloud_dataset()
    if osp.isfile(dataset_name):
        return load_synthetic_dataset(dataset_name)
    raise RuntimeError(f'Unknown dataset {dataset_name}. Please provide path to synthetic dataset file or correctly write real dataset name')
