

import numpy as np
from utils import Givens2Matrix, QRGivens, eigh_with_fixed_direction_range, find_closest_spd
from python_example import Givens2Matrix_double as Givens2Matrix
from python_example import QRGivens_double as QRGivens
from copy import deepcopy
from sklearn.mixture import GaussianMixture


class EigenParticle:
    def __init__(
        self, 
        n_components,
        data_dim,
        amplitude,
        weights,
        means,
        prec_matrix_list,
        data,
        means_coef,
        eig_val_max=1
    ):
        """
        Particle
        """
        self.data_dim = data_dim
        self.n_components = n_components
        self.amplitude = amplitude
        self.data = data

        self.means_coef = means_coef
        self.eig_val_max = eig_val_max
        
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
                'means' : means + np.random.normal(0, self.means_coef * np.mean(means), size=(n_components, data_dim)),
                'givens_angles' : np.random.uniform(-np.pi, np.pi, size=(n_components, int(data_dim * (data_dim - 1) / 2)))
            }
            #print('Eigvals: ', [np.mean(np.linalg.eigvals(item)) for item in prec_matrix_list], ' vs ', eig_val_max)
            
            self.position['eigenvalues_prec'] = np.array([np.random.uniform(0, np.mean(eig_val_max), size=data_dim) for i in range(self.n_components)])
        else:
            # decompose prec matrix list
            for i in range(n_components):
                eigenvalues, v = eigh_with_fixed_direction_range(cov_matrix_list[i])
                # eigenvalues, v = np.linalg.eigh(cov_matrix_list[i])
                givens_rotations = QRGivens(v)
                
                self.position['eigenvalues_prec'][i] = eigenvalues
                self.position['givens_angles'][i] = givens_rotations
            
        self.keys = self.position.keys()

        self.trajectory = [self.position]
        self.person_best = self.position
        self.global_best = self.position
        self.person_best_fitness_score = -np.inf

    def mutate(self):
        mutation_coef = 0.01
        self.position['means'] += np.random.normal(0, mutation_coef * self.means_coef * np.mean(self.position['means'].mean()), size=(self.n_components, self.data_dim))
        self.position['givens_angles'] += mutation_coef * np.random.uniform(-np.pi, np.pi, size=(self.n_components, int(self.data_dim * (self.data_dim - 1) / 2)))
        self.position['eigenvalues_prec'] += np.array([np.random.uniform(mutation_coef * np.mean(self.eig_val_max), mutation_coef * np.mean(self.eig_val_max), size=self.data_dim) for i in range(self.n_components)])

    def diversity(self, reference, threshold=1e-1):
        fraction = 0.01
        delta = np.abs(self.position['eigenvalues_prec'] - reference.position['eigenvalues_prec']).mean()
        return delta / self.position['eigenvalues_prec'].mean() > fraction

    def get_cov_matrices(self):
        ret_val = []
        for k in range(self.n_components):
            eigvals = self.position['eigenvalues_prec'][k]
            givens_rotations = self.position['givens_angles'][k]
            v = Givens2Matrix(np.expand_dims(givens_rotations, axis=1))
            cov_matr = v @ np.diag(eigvals) @ v.T
            ret_val.append(cov_matr)

        return self.best_prec
    
    def reorder_wrt(self, reference):

        for k in range(self.n_components):  
            eigen_in = self.position['eigenvalues_prec'][k]
            eigen_ref = reference['eigenvalues_prec'][k]
            from python_example import Givens2Matrix_double as Givens2Matrix

            v_ref = np.array(Givens2Matrix(np.expand_dims(reference['givens_angles'][k], axis=1)))
            v_in = np.array(Givens2Matrix(np.expand_dims(self.position['givens_angles'][k], axis=1)))

            eigen_res = []
            v_res = np.zeros_like(v_ref)
        
            untaken_inx = [i for i in range(self.data_dim)]
            permutation = np.zeros([self.data_dim], dtype=np.int32)
            for i in range(self.data_dim):
                i_opt = untaken_inx[np.argmax([abs(np.dot(v_ref[:, i], v_in[:, j])) for j in untaken_inx])]
                untaken_inx.remove(i_opt)
                permutation[i] = i_opt
                
                v_res[i] = deepcopy(v_in[:, i_opt])
                eigen_res.append(deepcopy(eigen_in[i_opt]))

            v_res = v_in[:, permutation]
            eigen_res = eigen_in[permutation]
            
            # print('Norm: ', np.linalg.norm( (v_res @  v_res.T) - (v_in @ v_in.T) ), 'With eig',  np.linalg.norm( (v_res @ np.diag(np.array(eigen_res)) @ v_res.T) - (v_in @ np.diag(eigen_in) @ v_in.T) ))
            
            self.position['eigenvalues_prec'][k] = np.array(eigen_res)

            # from utils import QRGivens
            givens_rotations = QRGivens(np.array(v_res)).squeeze()
            
            self.position['givens_angles'][k] = np.array(givens_rotations)

    def calculate_LL(self, data):
        if self.random_init:
            prec_matrcies = np.zeros_like(self.basic_prec_matr)
            # prec_matrcies = self.basic_prec_matr

            # construct prec matr addition
            for i in range(self.n_components):
                v = Givens2Matrix(np.expand_dims(self.position['givens_angles'][i], axis=1))
                addition = v @ np.diag(self.position['eigenvalues_prec'][i]) @ v.T
                prec_matrcies[i] = find_closest_spd(self.basic_prec_matr[i] + addition)
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
    
    def _calculate_LL_by_pos(self, data, position):
        if self.random_init:
            prec_matrcies = np.zeros_like(self.basic_prec_matr)

            # prec_matrcies = self.basic_prec_matr
            # construct prec matr addition
            for i in range(self.n_components):
                v = Givens2Matrix(np.expand_dims(position['givens_angles'][i], axis=1))
                addition = v @ np.diag(position['eigenvalues_prec'][i]) @ v.T
                prec_matrcies[i] = find_closest_spd(self.basic_prec_matr[i] + addition)
        else:
            cov_matr_list = self.get_cov_matrices()
            prec_matrcies = [np.linalg.inv(cov_matr) for cov_matr in cov_matr_list]
            
        gmm = GaussianMixture(n_components=position['weights'].shape[0], covariance_type='full', weights_init=position['weights'], means_init=position['means'], precisions_init=prec_matrcies, max_iter=100)
        
        cholesky = np.zeros_like(prec_matrcies)
        
        for i in range(self.n_components):
            cholesky[i] = np.linalg.cholesky(prec_matrcies[i])

        gmm.weights_ = position['weights']
        gmm.means_ = position['means']
        gmm.precisions_cholesky_ = cholesky
        return gmm.score(data)
        
    def run_em(self, data, T2):

        if self.random_init:
            prec_matrcies = np.zeros_like(self.basic_prec_matr)
            # construct prec matr addition
            for i in range(self.n_components):

                v = Givens2Matrix(np.expand_dims(self.position['givens_angles'][i], axis=1))
                addition = v @ np.diag(self.position['eigenvalues_prec'][i]) @ v.T
                prec_matrcies[i] = find_closest_spd(self.basic_prec_matr[i] + addition)
        else:
            prec_matrcies = self.get_cov_matrices()

        # print(prec_matrcies.shape, self.position['means'].shape)
        # print(data.shape)
        gmm = GaussianMixture(n_components=self.position['weights'].shape[0], covariance_type='full', weights_init=self.position['weights'], means_init=self.position['means'], precisions_init=prec_matrcies, max_iter=T2,  verbose=0, verbose_interval=1)
        gmm.fit(data)
        weights = gmm.weights_
        means = gmm.means_
        precisions = gmm.precisions_
        self.best_prec = deepcopy(precisions)

        for i in range(self.n_components):
            precisions[i] = precisions[i] - self.basic_prec_matr[i]
        
        # self.position['weights'] = weights
        self.position['means'] = means

        for i in range(self.n_components):
            eigenvalues, v = eigh_with_fixed_direction_range(precisions[i])
            # eigenvalues, v = np.linalg.eigh(cov_matrix_list[i])
            givens_rotations = QRGivens(v).squeeze()
            
            self.position['eigenvalues_prec'][i] = eigenvalues
            self.position['givens_angles'][i] = givens_rotations

        if gmm.score(data) > self.person_best_fitness_score:
            self.person_best_fitness_score = gmm.score(data)
            self.person_best = self.position
        
        return gmm.score(data)

    def set_params_after_em(self, weights, means, precisions, ll):
        self.best_prec = deepcopy(precisions)

        for i in range(self.n_components):
            precisions[i] = find_closest_spd(precisions[i] - self.basic_prec_matr[i])
        
        self.position['weights'] = weights
        self.position['means'] = means

        for i in range(self.n_components):
            eigenvalues, v = eigh_with_fixed_direction_range(precisions[i])
            givens_rotations = QRGivens(v).squeeze()
            
            self.position['eigenvalues_prec'][i] = eigenvalues
            self.position['givens_angles'][i] = givens_rotations

        if ll > self.person_best_fitness_score:
            self.person_best_fitness_score = ll
            self.person_best = self.position

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

        if self.calculate_LL(self.data) > self._calculate_LL_by_pos(self.data, self.person_best):
            print(self.calculate_LL(self.data), self._calculate_LL_by_pos(self.data, self.person_best))
            self.position = self.person_best
