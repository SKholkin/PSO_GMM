
import numpy as np
from utils import Givens2Matrix, QRGivens, eigh_with_fixed_direction_range
from copy import deepcopy
from sklearn.mixture import GaussianMixture

class Particle():
    def __init__(
        self, 
        n_components,
        data_dim, 
        rank, 
        amplitude,
        init_scale,
        weights,
    ):
        """
        Particle
        """
        self.amplitude = amplitude
        self.keys = ['weights', 'delta_mean', 'delta_diag_prec', 'delta_param_prec', 'delta_param_prec_eigval']
        
        self.velocity = { 
            'weights' : np.zeros(n_components),
            'delta_mean' : np.zeros((n_components, data_dim)),
            'delta_diag_prec' : np.zeros((n_components, data_dim)),
            'delta_param_prec' : np.zeros((n_components, rank, data_dim)),
            'delta_param_prec_eigval' : np.zeros((n_components, rank))
        }

        self.position = {
            'weights' : weights,
            'delta_mean' : np.random.normal(0, init_scale, size=(n_components, data_dim)),
            'delta_diag_prec' : np.random.normal(0, init_scale, size=(n_components, data_dim)),
            'delta_param_prec' : np.random.normal(0, init_scale, size=(n_components, rank, data_dim)),
            'delta_param_prec_eigval' : np.random.normal(0, init_scale, size=(n_components, rank))
        }
        
        self.trajectory = [self.position]
        self.person_best = self.position
        self.global_best = self.position
        self.person_best_fitness_score = -np.inf
        
        
    def step(self, c_1, c_2, r_1, r_2):
        for key in self.keys:
            self.velocity[key] = (
                c_1 * r_1[key] * (self.person_best[key] - self.position[key]) + 
                c_2 * r_2[key] * (self.global_best[key] - self.position[key])
            )
            self.position[key] += self.amplitude * self.velocity[key]
        self.trajectory.append(self.position)



class EigenParticle:
    def __init__(
        self, 
        n_components,
        data_dim,
        amplitude,
        weights,
        means,
        prec_matrix_list,
        init_std=0.001,
        eig_val_max=1
    ):
        """
        Particle
        """
        self.data_dim = data_dim
        self.n_components = n_components
        self.amplitude = amplitude
        
        self.velocity = { 
            'weights' : np.zeros(n_components),
            'means' : np.zeros((n_components, data_dim)),
            'eigenvalues_prec' : np.zeros((n_components, data_dim)),
            'givens_angles' : np.zeros((n_components, int(data_dim * (data_dim - 1) / 2)))
        }

        self.position = {
            'weights' : weights,
            'means' : means,
            'eigenvalues_prec' : np.zeros((n_components, data_dim)),
            'givens_angles' : np.zeros((n_components, int(data_dim * (data_dim - 1) / 2)))
        }


        self.random_init = True
        if self.random_init:
            self.basic_prec_matr = prec_matrix_list
            self.position = {
                'weights' : weights,
                'means' : means + np.random.normal(0, init_std, size=(n_components, data_dim)),
                'eigenvalues_prec' : np.random.uniform(1, eig_val_max, size=(n_components, data_dim)),
                'givens_angles' : np.random.uniform(-np.pi, np.pi, size=(n_components, int(data_dim * (data_dim - 1) / 2)))
            }
        else:
            # decompose covariance matrix
            for i in range(n_components):
                eigenvalues, v = eigh_with_fixed_direction_range(cov_matrix_list[i])
                # eigenvalues, v = np.linalg.eigh(cov_matrix_list[i])
                q, r, givens_rotations = QRGivens(v)
                
                self.position['eigenvalues_prec'][i] = eigenvalues
                self.position['givens_angles'][i] = givens_rotations
            
        self.keys = self.position.keys()

        self.trajectory = [self.position]
        self.person_best = self.position
        self.global_best = self.position
        self.person_best_fitness_score = -np.inf

    def get_cov_matrices(self):
        ret_val = []
        for k in range(self.n_components):
            eigvals = self.position['eigenvalues_prec'][k]
            givens_rotations = self.position['givens_angles'][k]
            v = Givens2Matrix(givens_rotations)
            cov_matr = v @ np.diag(eigvals) @ v.T
            ret_val.append(cov_matr)

        return ret_val

    def reorder_wrt(self, reference):

        for k in range(self.n_components):
            eigen_in = self.position['eigenvalues_prec'][k]
            eigen_ref = reference.position['eigenvalues_prec'][k]
            v_ref = np.array(Givens2Matrix(reference.position['givens_angles'][k]))
            v_in = np.array(Givens2Matrix(self.position['givens_angles'][k]))

            eigen_res = []
            v_res = np.zeros_like(v_ref)
        
            for i in range(self.data_dim):
                i_opt = np.argmax([abs(np.dot(v_ref[:, i], v_in[:, j])) for j in range(self.data_dim)])
                v_res[:, i] = v_in[:, i_opt]
                eigen_res.append(eigen_in[i_opt])
            
            self.position['eigenvalues_prec'][k] = np.array(eigen_res)

            q, r, givens_rotations = QRGivens(np.array(v_res))

            self.position['givens_angles'][k] = np.array(givens_rotations)

    def calculate_LL(self, data):
        if self.random_init:
            prec_matrcies = np.zeros_like(self.basic_prec_matr)
            # construct prec matr addition
            for i in range(self.n_components):
                v = Givens2Matrix(self.position['givens_angles'][i])
                addition = v @ np.diag(self.position['eigenvalues_prec'][i]) @ v.T
                # print(np.linalg.norm(self.basic_prec_matr[i]), np.linalg.norm(addition))
                prec_matrcies[i] = self.basic_prec_matr[i] + addition
        else:
            cov_matr_list = self.get_cov_matrices()
            prec_matrcies = [np.linalg.inv(cov_matr) for cov_matr in cov_matr_list]
        gmm = GaussianMixture(n_components=self.position['weights'].shape[0], covariance_type='full', weights_init=self.position['weights'], means_init=self.position['means'], precisions_init=prec_matrcies, max_iter=100)

        cholesky = np.zeros_like(prec_matrcies)
        
        for i in range(self.n_components):
            cholesky[i] = np.linalg.cholesky(prec_matrcies[i])

        gmm.weights_ = self.position['weights']
        gmm.means_ = self.position['means']
        gmm.precisions_cholesky_ = cholesky
        return gmm.score(data)

    def run_em(self, data):
        T2 = 200

        if self.random_init:
            prec_matrcies = np.zeros_like(self.basic_prec_matr)
            # construct prec matr addition
            for i in range(self.n_components):
                v = Givens2Matrix(self.position['givens_angles'][i])
                addition = v @ np.diag(self.position['eigenvalues_prec'][i]) @ v.T
                prec_matrcies[i] = self.basic_prec_matr[i] + addition
        else:
            prec_matrcies = self.get_cov_matrices()

        gmm = GaussianMixture(n_components=self.position['weights'].shape[0], covariance_type='full', weights_init=self.position['weights'], means_init=self.position['means'], precisions_init=prec_matrcies, max_iter=T2)
        gmm.fit(data)
        weights = gmm.weights_
        means = gmm.means_
        precisions = gmm.precisions_
        return gmm.score(data)

    def step(self, c_1, c_2, r_1, r_2):
        # reordering of cur point w.r.t. personal best
        #self.reorder_wrt(self.person_best)
        # reordering of global best w.r.t. reposonal best
        #global_best = deepcopy(self.global_best)
        #global_best.reorder_wrt(self.person_best)

        for key in self.keys:
            self.velocity[key] = (
                c_1 * r_1[key] * (self.person_best[key] - self.position[key]) + 
                c_2 * r_2[key] * (self.global_best[key] - self.position[key])
            )
            self.position[key] += self.amplitude * self.velocity[key]
        self.trajectory.append(self.position)


if __name__ == '__main__':
    d = 100
    eigenvalues = np.random.uniform(1, 16, size=d)
    rotation_angels = np.random.uniform(0, np.pi, size=int(d * (d - 1) / 2))
    
    v = Givens2Matrix(rotation_angels)
    cov_matr = v @ np.diag(eigenvalues) @ v.T
    cov_matr_list = [cov_matr]
    means = np.random.uniform(0, 1, size=[1, d])

    particle_1 = EigenParticle(n_components=1, data_dim=d, amplitude=1, weights=np.random.uniform(0, 1, size=[1, d]), means=means, cov_matrix_list=cov_matr_list)

    particle_2 = EigenParticle(n_components=1, data_dim=d, amplitude=1, weights=np.random.uniform(0, 1, size=[1, d]), means=means, cov_matrix_list=[cov_matr])

    assert np.allclose(particle_1.get_cov_matrices()[0], particle_2.get_cov_matrices()[0])

    cov_matr_p_1 = particle_1.get_cov_matrices()[0]

    v = Givens2Matrix(particle_1.position['givens_angles'][0])
    
    v_perm = v.copy()
    eigenvalues_perm = particle_1.position['eigenvalues_prec'][0].copy()

    assert np.allclose(cov_matr_p_1, v @ np.diag(particle_1.position['eigenvalues_prec'][0]) @ v.T)

    perm = np.random.permutation(d)

    for i in range(d):
        v_perm[:, i] = v[:, perm[i]]
        eigenvalues_perm[i] = particle_1.position['eigenvalues_cov'][0][perm[i]]

    q, r, givens_angles_perm = QRGivens(v_perm)

    particle_1.position['givens_angles'][0] = givens_angles_perm
    particle_1.position['eigenvalues_cov'][0] = eigenvalues_perm

    assert not np.allclose(particle_1.position['givens_angles'][0], particle_2.position['givens_angles'][0])
    assert not np.allclose(particle_1.position['eigenvalues_cov'][0], particle_2.position['eigenvalues_cov'][0])

    print((cov_matr_p_1 - v_perm @ np.diag(eigenvalues_perm) @ v_perm.T).max())
    assert np.allclose(cov_matr_p_1, v_perm @ np.diag(eigenvalues_perm) @ v_perm.T)

    print((particle_1.get_cov_matrices()[0] - particle_2.get_cov_matrices()[0]).max())
    assert np.allclose(particle_1.get_cov_matrices()[0], particle_2.get_cov_matrices()[0], atol=1e-5)

    print(particle_1.position['givens_angles'][0], particle_2.position['givens_angles'][0])

    particle_1.reorder_wrt(particle_2)

    assert np.allclose(particle_1.position['eigenvalues_cov'][0], particle_2.position['eigenvalues_cov'][0])

    print(particle_1.position['givens_angles'][0], particle_2.position['givens_angles'][0])
    print((particle_1.position['givens_angles'][0] - particle_2.position['givens_angles'][0]).mean())

    assert np.allclose(particle_1.position['givens_angles'][0], particle_2.position['givens_angles'][0], atol=1e-05)

    print('Testing is succesfull!')
