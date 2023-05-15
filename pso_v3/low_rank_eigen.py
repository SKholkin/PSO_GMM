from pso_v3.base import PSO, Particle, GMMState
# from utils import Givens2Matrix, QRGivens
import numpy as np
from sklearn.mixture import GaussianMixture
from copy import deepcopy
from python_example import Givens2Matrix_double as Givens2Matrix
from python_example import QRGivens_double as QRGivens
from utils import eigh_with_fixed_direction_range, find_closest_spd, eigh, add_zeros_to_square


def get_n_givens_angles(dim, rank):
    return int(dim * (dim - 1) // 2) - (dim  - 1 - rank) * (dim - rank) // 2

class LowRankEigenParticle(Particle):
    def __init__(self, data, n_comp, init_state, intertia, r_1, r_2, rank, base_spd, mutation_coefs=(0.01, 0.1)):
        super(Particle, self).__init__()
        self.n_comp = n_comp
        self.dim = init_state.dim
        self.data = data
        self.intertia, self.r_1, self.r_2 = intertia, r_1, r_2
        self.rank = rank
        self.base_spd = base_spd
        
        self.number_of_givens_angles = get_n_givens_angles(self.dim, self.rank)

        self.velocity = {
            'weights' : np.zeros(self.n_comp),
            'means' : np.zeros((self.n_comp, self.dim)),
            'eigenvalues' : np.zeros((self.n_comp, self.rank)),
            'givens_angles' : np.zeros((self.n_comp, self.number_of_givens_angles))
        }

        self.position = self.gmm_state_to_low_rank_eigen(init_state)

        self.person_best = deepcopy(self.position)
        self.pb_score = self.score()
        self.last_opt_score = None

    def gmm_state_to_low_rank_eigen(self, gmm_state):
        
        eigenvalues, givens_angles = np.zeros([gmm_state.n_comp, self.rank]), np.zeros([gmm_state.n_comp, self.number_of_givens_angles])
        spd_matrices = gmm_state.spd_matrices - self.base_spd

        for i in range(self.n_comp):

            eivals, V = eigh_with_fixed_direction_range(spd_matrices[i])
            givens_rotations = LowRankEigenParticle.Matrix_to_Givens(V, self.dim, self.rank)
            
            eigenvalues[i] = eivals[:self.rank]
            givens_angles[i] = givens_rotations

            V_low_rank = LowRankEigenParticle.Givens_to_Matrix(givens_rotations, self.dim, self.rank)
            low_rank_spd_add = V_low_rank @ np.diag(eigenvalues[i]) @ V_low_rank.T
            # print(eigh(low_rank_spd_add)[0], ' vs ', eigh(spd_matrices[i])[0])

        return {
            'weights' : gmm_state.weights,
            'means' : gmm_state.means,
            'eigenvalues' : eigenvalues,
            'givens_angles' : givens_angles
            }

    @staticmethod
    def Givens_to_Matrix(givens_rotations, dim, rank):
        fn_inputs = np.zeros([int(dim * (dim - 1) // 2)])

        counter = 0
        for i in range(0, dim - 1):
            line_begin = int(i + 1) * i // 2
            for j in range(i + 1):
                if j < rank:
                    fn_inputs[line_begin + j] = givens_rotations[counter]
                    counter += 1
        ret_val = Givens2Matrix(np.expand_dims(fn_inputs, axis=1))
        ret_val[:, rank:] = np.zeros([ret_val.shape[0], dim - rank])
        return ret_val[:, :rank]
    
    @staticmethod
    def Matrix_to_Givens(V, dim, rank):
        V_low_rank = np.zeros_like(V)
        V_low_rank[:, :rank] = V[:, :rank]
        givens_rotations = QRGivens(np.array(V_low_rank)).squeeze()

        ret_val = np.zeros([get_n_givens_angles(dim, rank)])
        counter = 0
        for i in range(0, dim - 1):
            line_begin = int(i + 1) * i // 2
            for j in range(i + 1):
                if j < rank:
                    ret_val[counter] = givens_rotations[line_begin + j]
                    counter += 1
        
        return ret_val

    def get_prec_matrices(self, position):
        ret_val = []
        for k in range(self.n_comp):
            eigvals = position['eigenvalues'][k]
            givens_rotations = position['givens_angles'][k]
            v = LowRankEigenParticle.Givens_to_Matrix(givens_rotations, self.dim, self.rank)
            cov_matr = v @ np.diag(eigvals) @ v.T
            ret_val.append(cov_matr)

        return np.array([find_closest_spd(self.base_spd[i] + np.array(ret_val[i])) for i in range((self.base_spd + np.array(ret_val)).shape[0])])

    def opt_step(self):
        prec_matrcies = self.get_prec_matrices(self.position)
        gmm = GaussianMixture(n_components=self.position['weights'].shape[0], covariance_type='full', weights_init=self.position['weights'], means_init=self.position['means'], precisions_init=prec_matrcies, verbose=0, verbose_interval=1)
        gmm.fit(self.data)
        gmmstate = GMMState.from_gmm(gmm)
        # print(f'Opt step: {gmm.score(self.data)}')
        self.last_opt_score = gmm.score(self.data)

        self.position = self.gmm_state_to_low_rank_eigen(gmmstate)

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
        for k in range(prec_matrcies.shape[0]):
            values, vectors = eigh(prec_matrcies[k])

        gmm = GaussianMixture(n_components=self.position['weights'].shape[0], covariance_type='full', weights_init=self.position['weights'], means_init=self.position['means'], precisions_init=prec_matrcies, verbose=0, verbose_interval=1)

        gmm.weights_ = self.position['weights']
        gmm.means_ = self.position['means']
        gmm.precisions_cholesky_ = LowRankEigenParticle.get_prec_cholesky(prec_matrcies)

        return gmm.score(self.data)

    def reorder_wrt(self, to_reoder, reference):
        n_comp = to_reoder['weights'].shape[0]
        dim = to_reoder['means'].shape[1]
        for k in range(n_comp):  
            eigen_in = to_reoder['eigenvalues'][k]
            eigen_ref = reference['eigenvalues'][k]

            v_ref = np.array(LowRankEigenParticle.Givens_to_Matrix(reference['givens_angles'][k], self.dim, self.rank))
            v_in = np.array(LowRankEigenParticle.Givens_to_Matrix(to_reoder['givens_angles'][k], self.dim, self.rank))
            
            eigen_res = []
            v_res = np.zeros_like(v_ref)

            untaken_inx = [i for i in range(self.rank)]
            permutation = np.zeros([self.rank], dtype=np.int32)
            for i in range(self.rank):
                i_opt = untaken_inx[np.argmax([abs(np.dot(v_ref[:, i], v_in[:, j])) for j in untaken_inx])]
                untaken_inx.remove(i_opt)
                permutation[i] = i_opt

                eigen_res.append(deepcopy(eigen_in[i_opt]))

            v_res = v_in[:, permutation]
            eigen_res = eigen_in[permutation]
            
            to_reoder['eigenvalues'][k] = np.array(eigen_res)

            v_res = add_zeros_to_square(np.array(v_res))

            givens_rotations = LowRankEigenParticle.Matrix_to_Givens(v_res, self.dim, self.rank).squeeze()
            
            to_reoder['givens_angles'][k] = np.array(givens_rotations)

        return to_reoder

class LowRankEigenPSO(PSO):
    def __init__(self, particles, n_steps):
        self.best_opt_score = -np.inf
        super(LowRankEigenPSO, self).__init__(particles, n_steps)

    def check_gb(self):
        print('check gb')
        self.since_last_gb_change += 1
        for particle in self.particles:
            score = particle.score()
            if score > self.gb_score:
                self.gb_score = score
                self.gb_position = particle.position.copy()
                self.since_last_gb_change = 0

            if particle.last_opt_score is not None:
                if particle.last_opt_score > self.best_opt_score:
                    self.best_opt_score = particle.last_opt_score
            
        gb = self.gb_position.copy()

        for particle in self.particles:
            particle.set_gb(gb)

            # reorder wrt new GB
            particle.position = particle.reorder_wrt(particle.position, gb)
            particle.person_best = particle.reorder_wrt(particle.person_best, gb)

    def get_best_score(self):
        return  self.best_opt_score

def gmm_init_states(data, n_comp, n_particles):
    ret_val = []
    for i in range(n_particles):
        gmm = GaussianMixture(n_comp, covariance_type='full')
        gmm.fit(data)
        init_state = GMMState.from_gmm(gmm)
        ret_val.append(init_state)

    return ret_val

def gmm_scatter_around_best_init_states(data, n_comp, n_particles):
    raise NotImplementedError

def get_base_spd(list_of_gmm):

    n_comp = list_of_gmm[0].n_comp

    base_spd = np.zeros_like(list_of_gmm[0].spd_matrices)

    for n in range(n_comp):
        sum_of_spd = np.zeros_like(list_of_gmm[0].spd_matrices[n])

        for gmm in list_of_gmm:
            sum_of_spd += gmm.spd_matrices[n]

        base_spd[n] = sum_of_spd / n_comp

    return base_spd

if __name__ == '__main__':
    pass
    from sklearn.datasets import make_spd_matrix
    
    dim, rank = 30, 15
    spd = make_spd_matrix(dim)

    values, vectors = np.linalg.eigh(spd)

    print(values)

    print(get_n_givens_angles(dim, rank))

    givens_angles = LowRankEigenParticle.Matrix_to_Givens(vectors, get_n_givens_angles(dim, rank))

    spd_low_rank = LowRankEigenParticle.Givens_to_Matrix(givens_angles, dim)

    np.linalg.rank(spd_low_rank)
