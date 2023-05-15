from types import SimpleNamespace
import numpy as np
from copy import deepcopy
from python_example import Givens2Matrix_double as Givens2Matrix
from python_example import QRGivens_double as QRGivens
from utils import eigh_with_fixed_direction_range
from sklearn.mixture import GaussianMixture
from pso_v3.eigen_pso import EigenParticle

from pso_v3.base import GMMState

class GMMStateOld:
    def __init__(self, gmm) -> None:
        self.means = gmm.means_.astype('float64')
        self.weights = gmm.weights_.astype('float64')
        self.precisions = gmm.precisions_.astype('float64')
        self.n_comp = self.means.shape[0]

    def zeros_like(self):
        gmm_state_dict = {'means_': np.zeros_like(self.means),
                          'weights_': np.zeros_like(self.weights), 'precisions_': self.precisions}
        return GMMStateOld(SimpleNamespace(**gmm_state_dict))
    
    def __add__(self, gmm_2):
        self = deepcopy(self)
        self.means += gmm_2.means
        self.weights += gmm_2.weights
        self.precisions += gmm_2.precisions
        return self

class GMMStateSpectral:
    def __init__(self, means, weights, eigvals, rotations) -> None:
        self.means = means
        self.weights = weights
        self.eigvals = eigvals
        self.rotations = rotations
        self.n_comp, self.d = means.shape[0], means.shape[1]

    def scatter(self, std):
        delta_means = np.random.normal(0, std * np.mean(self.means), size=(self.n_comp, self.d))
        delta_givens =  np.random.uniform(-np.pi * std, std * np.pi, size=(self.n_comp, int(self.d * (self.d - 1) / 2)))

        eig_val_mean = [np.mean(self.eigvals[i]) for i in range(self.n_comp)]

        delta_eigvals = np.random.uniform(-std * np.mean(eig_val_mean), std * np.mean(eig_val_mean), size=(self.n_comp, self.d))

        self.means += delta_means
        self.rotations += delta_givens
        self.eigvals += delta_eigvals
        self.eigvals = (self.eigvals > 0) * self.eigvals + 1e-04
        # print(self.eigvals)


def to_linear(gmm_state: GMMStateOld):
    n_comp = gmm_state.precisions.shape[0]
    d = gmm_state.precisions.shape[1]

    eigvals = np.zeros([n_comp, d])
    rotations = np.zeros([n_comp, int(d * (d - 1) / 2)])

    for i in range(gmm_state.precisions.shape[0]):
        prec = gmm_state.precisions[i]

        eigenvalues, v = eigh_with_fixed_direction_range(prec)
        givens_rotations = QRGivens(v).squeeze()
        
        eigvals[i] = eigenvalues
        rotations[i] = givens_rotations
    gmm_spec = GMMStateSpectral(gmm_state.means, gmm_state.weights, eigvals, rotations)
    return gmm_spec
    
def to_manifold(gmm_spec_state: GMMStateSpectral) -> GMMStateOld:

    n_comp, d = gmm_spec_state.means.shape[0], gmm_spec_state.means.shape[1]

    prec_matrices = np.zeros([n_comp, d, d])
    for k in range(n_comp):
            eigvals = gmm_spec_state.eigvals[k]
            givens_rotations = gmm_spec_state.rotations[k]
            v = Givens2Matrix(np.expand_dims(givens_rotations, axis=1))
            spd_matr = v @ np.diag(eigvals) @ v.T
            prec_matrices[k] = spd_matr

    gmm_state_dict = {'means_': gmm_spec_state.means,
                        'weights_': gmm_spec_state.weights, 'precisions_': prec_matrices}
    
    return GMMStateOld(SimpleNamespace(**gmm_state_dict))

def gmm_scatter_around_best_init_states(data, n_comp, n_particles, std=0.01, gmm_states=None):
    if gmm_states is not None:
        states = [EigenParticle(data, n_comp, state, 0, 0, 0) for state in gmm_states]
        states = sorted(states, key=lambda x:-x.score())
        gmm = states[0].to_gmm()
    else:
        gmm = GaussianMixture(n_components=n_comp, covariance_type='full', n_init=n_particles, init_params='k-means++')
        gmm.fit(data)

    basic_score = gmm.score(data)
    print('Basic GMM score:', gmm.score(data))
    gmm_state = GMMStateOld(gmm)
    basic_state = deepcopy(gmm_state)
    ret_val = []
    for i in range(n_particles):
        gmm_state_copy = deepcopy(gmm_state)
        gmm_spectral_state = to_linear(gmm_state_copy)
        gmm_spectral_state.scatter(std)

        ret_val.append(to_manifold(gmm_spectral_state))

    return [GMMState(item.weights, item.means, item.precisions) for item in ret_val], basic_state.precisions

def get_best_gmm_state(states, data, n_comp):
    # data, n_comp, init_state, intertia, r_1, r_2
    particles = [EigenParticle(data, n_comp, state, 0, 0, 0) for state in states]
    particles = sorted(particles, key=lambda x:-x.score())
    return GMMState.from_gmm(particles[0].to_gmm()), particles[0].to_gmm().score(data)
