from pso_v3.base import PSO, Particle, GMMState
# from utils import Givens2Matrix, QRGivens
import numpy as np
from sklearn.mixture import GaussianMixture
from copy import deepcopy
from python_example import Givens2Matrix_double as Givens2Matrix
from python_example import QRGivens_double as QRGivens



class EigenParticle(Particle):
    def __init__(self, data, n_comp, init_state, intertia, r_1, r_2, mutation_coefs=(0.01, 0.1)):
        super(Particle, self).__init__()
        self.n_comp = n_comp
        self.dim = init_state.dim
        self.data = data
        self.intertia, self.r_1, self.r_2 = intertia, r_1, r_2

        self.velocity = { 
            'weights' : np.zeros(self.n_comp),
            'means' : np.zeros((self.n_comp, self.dim)),
            'eigenvalues' : np.zeros((self.n_comp, self.dim)),
            'givens_angles' : np.zeros((self.n_comp, int(self.dim * (self.dim - 1) / 2)))
        }

        self.position = init_state.to_eigen()

        self.person_best = deepcopy(self.position)
        self.pb_score = self.score()

    def get_prec_matrices(self, position):
        ret_val = []
        for k in range(self.n_comp):
            eigvals = position['eigenvalues'][k]
            givens_rotations = position['givens_angles'][k]
            v = Givens2Matrix(np.expand_dims(givens_rotations, axis=1))
            cov_matr = v @ np.diag(eigvals) @ v.T
            ret_val.append(cov_matr)

        return np.array(ret_val)

    def opt_step(self):
        prec_matrcies = self.get_prec_matrices(self.position)
        gmm = GaussianMixture(n_components=self.position['weights'].shape[0], covariance_type='full', weights_init=self.position['weights'], means_init=self.position['means'], precisions_init=prec_matrcies, verbose=0, verbose_interval=1)
        gmm.fit(self.data)
        gmmstate = GMMState.from_gmm(gmm)
        self.position = gmmstate.to_eigen()

    def to_gmm(self):
        prec_matrcies = self.get_prec_matrices(self.position)
        gmm = GaussianMixture(n_components=self.position['weights'].shape[0], covariance_type='full', weights_init=self.position['weights'], means_init=self.position['means'], precisions_init=prec_matrcies, verbose=0, verbose_interval=1)
        gmm.fit(self.data)
        return gmm

    def pso_step(self):

        c_1, c_2 = np.random.uniform(), np.random.uniform()
        for key in self.position.keys():
            self.velocity[key] = (
                c_1 * self.r_1 * (self.person_best[key] - self.position[key]) + 
                c_2 * self.r_2 * (self.global_best[key] - self.position[key]) + 
                self.intertia * self.velocity[key] 
            )

            self.position[key] += self.velocity[key]
        self.position['weights'] = (self.position['weights'] - np.min(self.position['weights']) + 1e-4) 
        self.position['weights'] = self.position['weights'] / np.sum(self.position['weights'])
        self.position['eigenvalues'] = np.absolute(self.position['eigenvalues'])

        # print(self.position['eigenvalues'])
        # update personal best
        self.check_pb()
    
    def set_gb(self, gb):
        self.global_best = deepcopy(gb)

    def check_pb(self):
        if self.score() > self.pb_score:
            self.person_best = deepcopy(self.position)
            self.pb_score = self.score()

    @staticmethod
    def get_prec_cholesky(prec_matrcies):

        cholesky = np.zeros_like(prec_matrcies)
        for i in range(prec_matrcies.shape[0]):
            cholesky[i] = np.linalg.cholesky(prec_matrcies[i])

        return cholesky

    def score(self):
        prec_matrcies = self.get_prec_matrices(self.position)
        gmm = GaussianMixture(n_components=self.position['weights'].shape[0], covariance_type='full', weights_init=self.position['weights'], means_init=self.position['means'], precisions_init=prec_matrcies, verbose=0, verbose_interval=1)

        gmm.weights_ = self.position['weights']
        gmm.means_ = self.position['means']
        gmm.precisions_cholesky_ = EigenParticle.get_prec_cholesky(prec_matrcies)

        return gmm.score(self.data)

    @staticmethod
    def reorder_wrt(to_reoder, reference):
        n_comp = to_reoder['weights'].shape[0]
        dim = to_reoder['means'].shape[1]
        for k in range(n_comp):  
            eigen_in = to_reoder['eigenvalues'][k]
            eigen_ref = reference['eigenvalues'][k]

            v_ref = np.array(Givens2Matrix(np.expand_dims(reference['givens_angles'][k], axis=1)))
            v_in = np.array(Givens2Matrix(np.expand_dims(to_reoder['givens_angles'][k], axis=1)))

            eigen_res = []
            v_res = np.zeros_like(v_ref)
        
            untaken_inx = [i for i in range(dim)]
            permutation = np.zeros([dim], dtype=np.int32)
            for i in range(dim):
                i_opt = untaken_inx[np.argmax([abs(np.dot(v_ref[:, i], v_in[:, j])) for j in untaken_inx])]
                untaken_inx.remove(i_opt)
                permutation[i] = i_opt
                
                v_res[i] = deepcopy(v_in[:, i_opt])
                eigen_res.append(deepcopy(eigen_in[i_opt]))

            v_res = v_in[:, permutation]
            eigen_res = eigen_in[permutation]
            
            
            to_reoder['eigenvalues'][k] = np.array(eigen_res)

            givens_rotations = QRGivens(np.array(v_res)).squeeze()
            
            to_reoder['givens_angles'][k] = np.array(givens_rotations)

        return to_reoder

class EigenPSO(PSO):
    def __init__(self, particles, n_steps):
        super(EigenPSO, self).__init__(particles, n_steps)

    def check_gb(self):
        self.since_last_gb_change += 1
        for particle in self.particles:
            score = particle.score()
            if score > self.gb_score:
                self.gb_score = score
                self.gb_position = particle.position.copy()
                self.since_last_gb_change = 0
            
        gb = self.gb_position.copy()
        for particle in self.particles:
            particle.set_gb(gb)
            
            # reorder wrt new GB
            particle.position = EigenParticle.reorder_wrt(particle.position, gb)
            particle.person_best = EigenParticle.reorder_wrt(particle.person_best, gb)

def gmm_init_states(data, n_comp, n_particles):
    ret_val = []
    scores = []
    for i in range(n_particles):
        gmm = GaussianMixture(n_comp, covariance_type='full')
        gmm.fit(data)
        scores.append(gmm.score(data))
        init_state = GMMState.from_gmm(gmm)
        ret_val.append(init_state)

    return ret_val, scores
