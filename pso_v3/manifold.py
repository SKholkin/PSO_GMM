
import numpy as np

from numpy.linalg import eigh

class SPDManifold:
    def __init__(self) -> None:
        pass

    @staticmethod
    def Log(Sigma, X):
        sqrt_sigma, inv_sqrt_sigma = SPDManifold.spd_pos_neg_sqrt(Sigma)
        middle = SPDManifold.spd_log(inv_sqrt_sigma @ X @ inv_sqrt_sigma)
        return sqrt_sigma @ middle @ sqrt_sigma
    
    @staticmethod
    def Exp(Sigma, X):
        sqrt_sigma, inv_sqrt_sigma = SPDManifold.spd_pos_neg_sqrt(Sigma)
        middle = SPDManifold.spd_exp(inv_sqrt_sigma @ X @ inv_sqrt_sigma)
        return sqrt_sigma @ middle @ sqrt_sigma
    
    @staticmethod
    def spd_pos_neg_sqrt(Sigma):
        eigval, eigvec = eigh(Sigma)
        eigvals_neg_sqrt, eigvals_pos_sqrt = np.sqrt(eigval),  1 / np.sqrt(eigval)
        return eigvec @ np.diag(eigvals_pos_sqrt) @ eigvec.T, eigvec @ np.diag(eigvals_neg_sqrt) @ eigvec.T 
    
    @staticmethod
    def spd_exp(Sigma):
        eigval, eigvec = eigh(Sigma)
        exp_eigvals = np.exp(eigval)
        return eigvec @ np.diag(exp_eigvals) @ eigvec.T

    @staticmethod
    def spd_log(Sigma):
        eigval, eigvec = eigh(Sigma)
        log_eigvals = np.log(eigval)
        return eigvec @ np.diag(log_eigvals) @ eigvec.T
