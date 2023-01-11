from sklearn.preprocessing import MinMaxScaler
import os.path as osp
import pandas as pd
import numpy as np
from math import copysign, hypot

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


def _givens_rotation_matrix_entries(a, b):
    """Compute matrix entries for Givens rotation.[[cos(phi), -sin(phi)], [sin(phi), cos(phi)]]"""
    r = hypot(a, b)
    c = a/r
    s = -b/r

    return (c, s)


def QRGivens(A):
    if np.linalg.det(A) < 0:
        A = -A
    """Perform QR decomposition of matrix A using Givens rotation."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)
    phi_list = []

    # Iterate over lower triangular matrix.
    (rows, cols) = np.tril_indices(num_rows, -1, num_cols)
    i = 0
    for (row, col) in zip(rows, cols):
        i += 1

        # Compute Givens rotation matrix and
        # zero-out lower triangular matrix entries.
        (c, s) = _givens_rotation_matrix_entries(R[col, col], R[row, col])


        phi = np.arccos(c)
        # if sin(phi) < 0
        if s > 0:
            phi = -phi

        # Turning first element into 1 instead of -1
        if c * R[col, col] - s * R[row, col] < 0:
            phi = phi - np.pi
            c = -c
            s = -s

        G = np.identity(num_rows)
        G[col, col] = c
        G[row, row] = c
        phi_list.append(phi)
        G[row, col] = s
        G[col, row] = -s


        R = np.dot(G, R)

        Q = np.dot(Q, G.T)

    return Q, R, phi_list


def Givens2Matrix(phi_list):
    d  = int((1 + np.sqrt(1 + 8 * len(phi_list))) / 2)
    ret_val = np.eye(d)
    i = 0
    (rows, cols) = np.tril_indices(d, -1, d)
    for (row, col) in zip(rows, cols):
        
        c = np.cos(phi_list[i])
        s = -np.sin(phi_list[i])
        i += 1

        G = np.eye(d)
        G[[col, row], [col, row]] = c

        G[row, col] = s
        G[col, row] = -s

        ret_val = np.dot(ret_val, G.T)

        
    return ret_val
        
def eigh_with_fixed_direction_range(spd_matr):
    eigenvalues, v = np.linalg.eigh(spd_matr)

    base_vector = np.ones_like(v[0])
    for i in range(v.shape[0]):
        cos_phi = np.dot(base_vector, v[:, i])
        if cos_phi > 0:
            v[:, i] = -v[:, i]

    return eigenvalues, v
