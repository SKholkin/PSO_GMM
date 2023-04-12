from python_example import QRGivens_double, Givens2Matrix_double
from utils import QRGivens, Givens2Matrix


from sklearn.datasets import make_spd_matrix

dim = 5
A = make_spd_matrix(dim)
phi_list = QRGivens_double(A)
print(phi_list)
print(QRGivens(A)[2])
print(Givens2Matrix_double(phi_list))
print(Givens2Matrix(phi_list))
