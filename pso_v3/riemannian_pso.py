from pso_v3.base import PSO, Particle, GMMState
from utils import eigh_with_fixed_direction_range, find_closest_spd
import numpy as np
from sklearn.mixture import GaussianMixture
from copy import deepcopy
from python_example import Givens2Matrix_double as Givens2Matrix
from python_example import QRGivens_double as QRGivens



from pymanopt.manifolds import SymmetricPositiveDefinite
from pso_v3.manifold import SPDManifold
from pso_v3.eigen_pso import EigenParticle

class RiemannianPartilce(Particle):
    def __init__(self, data, n_comp, init_state, intertia, r_1, r_2, mutation_coefs=(0.01, 0.1)):
        self.data = data
        self.n_comp = n_comp
        self.intertia, self.r_1, self.r_2 = intertia, r_1, r_2
        self.dim = init_state.means.shape[1]
        self.pymanopt_manifold = SymmetricPositiveDefinite(self.dim)
        self.log, self.ret = SPDManifold.Log, self.pymanopt_manifold.retraction
        
        self.velocity = { 
            'weights' : np.zeros(self.n_comp),
            'means' : np.zeros((self.n_comp, self.dim)),
            'spd': np.zeros([self.n_comp, self.dim, self.dim])
        }

        # self.position = { 
        #     'weights' : init_state.weights,
        #     'means' : init_state.means,
        #     'spd': init_state.spd_matrices
        # }

        self.position = init_state.to_normal()

        self.person_best = deepcopy(self.position)
        self.pb_score = self.score()

    @staticmethod
    def log_spd(spd_matrices_base, spd_matrices_to_project, log_fn):
        ret_val = []
        for i in range(spd_matrices_to_project.shape[0]):
            ret_val.append(log_fn(spd_matrices_base[i], spd_matrices_to_project[i]))

        return np.array(ret_val)
    
    def set_gb(self, gb):
        self.global_best = deepcopy(gb)
    
    @staticmethod
    def ret_spd(spd_matrices_base, tangent_vectors, ret_fn):
        ret_val = []
        for i in range(tangent_vectors.shape[0]):
            ret_val.append(ret_fn(spd_matrices_base[i], tangent_vectors[i]))

        return np.array(ret_val)

    def score(self):
        prec_matrcies = self.position['spd']
        gmm = GaussianMixture(n_components=self.position['weights'].shape[0], covariance_type='full', weights_init=self.position['weights'], means_init=self.position['means'], precisions_init=prec_matrcies, verbose=0, verbose_interval=1)

        gmm.weights_ = self.position['weights']
        gmm.means_ = self.position['means']
        gmm.precisions_cholesky_ = EigenParticle.get_prec_cholesky(prec_matrcies)

        return gmm.score(self.data)

    def opt_step(self):
        gmm = GaussianMixture(n_components=self.position['weights'].shape[0], covariance_type='full', weights_init=self.position['weights'], means_init=self.position['means'], precisions_init=self.position['spd'], verbose=0, verbose_interval=1)
        gmm.fit(self.data)
        gmmstate = GMMState.from_gmm(gmm)
        self.position = gmmstate.to_normal()

    def pso_step(self):

        c_1, c_2 = np.random.uniform(), np.random.uniform()
        self.velocity['weights'] = (c_1 * self.r_1 * (self.person_best['weights'] - self.position['weights']) + 
                                    c_2 * self.r_2 * self.global_best['weights'] - self.position['weights'])
        self.velocity['means'] = (c_1 * self.r_1 * (self.person_best['means'] - self.position['means']) + 
                                    c_2 * self.r_2 * self.global_best['means'] - self.position['means'])

        self.position['means'] += self.velocity['means']

        self.position['weights'] += self.velocity['weights']
        self.position['weights'] = (self.position['weights'] - np.min(self.position['weights']) + 1e-4) 
        self.position['weights'] = self.position['weights'] / np.sum(self.position['weights'])

        velocity_projected = RiemannianPartilce.log_spd(self.position['spd'], self.velocity['spd'], self.log)
        pb_projected = RiemannianPartilce.log_spd(self.position['spd'], self.person_best['spd'], self.log)
        gb_projected = RiemannianPartilce.log_spd(self.position['spd'], self.global_best['spd'], self.log)
        position_projected =  RiemannianPartilce.log_spd(self.position['spd'], self.position['spd'], self.log)

        new_tangent_velocity_spd = c_1 * self.r_1 * (pb_projected  - position_projected) + c_2 * self.r_2 * (gb_projected - position_projected)

        position_new_tangent = position_projected + new_tangent_velocity_spd

        self.position['spd'] = RiemannianPartilce.ret_spd(self.position['spd'], position_new_tangent, self.ret)
