import numpy as np
from sklearn.mixture import GaussianMixture
from pathlib import Path
import json
from copy import deepcopy
from addict import Dict
import datetime
from particle import Particle, EigenParticle
from utils import find_closest_spd
import time


class PSOConfig(Dict):
    default_params_path = 'default_params.json'
    must_have_params = ["n_particles",  "amplitude",  "rank", "init_scale", "T2", "T1", "n_components", "n_iters"]
    @classmethod
    def from_json(cls, path):
        file_path = Path(path).resolve()
        with open(file_path) as f:
            loaded_json = json.load(f)
        PSOConfig.check_compliteness(loaded_json)
        return cls(loaded_json)

    @staticmethod
    def check_compliteness(loaded_json):
        for param_name in PSOConfig.must_have_params:
            if param_name not in loaded_json.keys():
                raise RuntimeError(f"Param {param_name} is not in config json file") 
        
class PSO():
    def __init__(
        self,
        data,
        config: PSOConfig,
        basic_gmm=None
    ):
        """
        PSO
        """
        self.config = config
        self.n_particles = config.n_particles
        self.amplitude = config.amplitude
        self.n_components = config.n_components
        self.rank = config.rank
        self.init_scale = config.init_scale
        self.T2 = config.T2
        self.T1 = config.T1
        self.data = data
        
        r_1 = 0.6
        r_2 = 0.8
        r_1_w = 0.42
        r_2_w = 0.57
            
        self.r_1 = {
            'weights' : r_1_w,
            'delta_mean' : r_1,
            'delta_diag_prec' : r_1,
            'delta_param_prec' : r_1,
            'delta_param_prec_eigval': r_1
        }
        
        self.r_2 = {
            'weights' : r_2_w,
            'delta_mean' : r_2,
            'delta_diag_prec' : r_2,
            'delta_param_prec' : r_2,
            'delta_param_prec_eigval': r_2
        }
        self.data_dim = data.shape[1]
        
        if basic_gmm is None:
            self.basic_weights, self.basic_means, self.basic_prec_matr, self.basic_gmm = self.basic_gmm_init()
        else:
            self.basic_weights, self.basic_means = basic_gmm.weights_, basic_gmm.means_

            self.basic_prec_matr = np.zeros_like(basic_gmm.precisions_cholesky_)
            for i in range(basic_gmm.precisions_cholesky_.shape[0]):
                self.basic_prec_matr[i] = basic_gmm.precisions_cholesky_[i] @ basic_gmm.precisions_cholesky_[i].T
            self.basic_gmm = basic_gmm

        self.particles = [Particle(self.n_components, self.data_dim, self.rank, self.amplitude, self.init_scale, self.basic_weights, self.basic_means)
                          for _ in range(self.n_particles)]
        self._init_global()
        

    def basic_gmm_init(self):
        
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', n_init=self.n_particles, max_iter=self.T2 * self.T1,  init_params='k-means++')
        gmm.fit(self.data)
        weights = gmm.weights_
        means = gmm.means_
        
        basic_prec_matr = np.zeros_like(gmm.precisions_cholesky_)
        for i in range(gmm.precisions_cholesky_.shape[0]):
            basic_prec_matr[i] = gmm.precisions_cholesky_[i] @ gmm.precisions_cholesky_[i].T
        self.basic_gmm_score = gmm.score(self.data)

        return weights, means, basic_prec_matr, gmm

    def calculate_log_likelihood_gmm(self, particle: Particle):
        weights = particle.position['weights']

        # means = deepcopy(self.basic_means) + particle.position['delta_mean']
        means = particle.position['delta_mean']

        # means = deepcopy(self.basic_means)

        prec_matr = deepcopy(self.basic_prec_matr)

        cholesky = np.zeros_like(prec_matr)
        
        for i in range(self.n_components):

            # prec_matr[i] += np.diag(particle.position['delta_diag_prec'][i] ** 2)

            # for k in range(self.config.rank):
            #     prec_matr[i] += particle.position['delta_param_prec_eigval'][i][k] * (particle.position['delta_param_prec'][i][k] @ particle.position['delta_param_prec'][i][k].T)
            prec_matr[i] = find_closest_spd(prec_matr[i])
            cholesky[i] = np.linalg.cholesky(prec_matr[i])

        gmm_init = GaussianMixture(n_components=weights.shape[0], covariance_type='full', weights_init=weights, means_init=means, max_iter=self.config.T2)
        gmm_init.weights_ = weights
        gmm_init.means_ = means
        gmm_init.precisions_cholesky_ = cholesky
        return gmm_init.score(self.data)
        
    def _init_global(self):
        fintess_scores = np.array([self.calculate_log_likelihood_gmm(particle) 
                           for particle in self.particles])
        best_id = np.argmin(fintess_scores)
        self.global_fitness_score = fintess_scores[best_id]
        self.global_best = self.particles[best_id].position
        
    def reinit_particles(self, init_scale=None, inplace=True):
        if init_scale is None:
            init_scale = self.init_scale
        particles = [Particle(self.n_components, self.data_dim, self.rank, self.amplitude, init_scale, self.basic_weights, self.basic_means) for _ in range(self.n_particles)]
        if inplace:
            self.particles = particles
        return particles

    def run_em_particles_fintess_score(self):
        score_list = []
        for particle in self.particles:
            weights = particle.position['weights']

            means = deepcopy(self.basic_means) + particle.position['delta_mean']

            prec_matr = deepcopy(self.basic_prec_matr)

            cholesky = np.zeros_like(prec_matr)
            
            for i in range(self.n_components):

                prec_matr[i] += np.diag(particle.position['delta_diag_prec'][i] ** 2)

                for k in range(self.config.rank):
                    prec_matr[i] += particle.position['delta_param_prec'][i][k] @ particle.position['delta_param_prec'][i][k].T

                cholesky[i] = np.linalg.cholesky(prec_matr[i])

            gmm_init = GaussianMixture(n_components=weights.shape[0], covariance_type='full', weights_init=weights, means_init=means, precisions_init=prec_matr, max_iter=self.config.T2)
            gmm_init.fit(self.data)
            score_list.append(gmm_init.score(self.data))
        return score_list

    def get_particles_fitness_scores(self):
        fintess_scores = np.array([self.calculate_log_likelihood_gmm(particle) 
                           for particle in self.particles])
        return fintess_scores

    def step(self):
        c_1 = np.random.uniform(0, 1)
        c_2 = np.random.uniform(0, 1)
        
        for particle in self.particles:
            particle.global_best = self.global_best
            particle.step(c_1, c_2, self.r_1, self.r_2)
            fintess_score = self.calculate_log_likelihood_gmm(particle)

            if fintess_score > particle.person_best_fitness_score:
                particle.person_best_fitness_score = fintess_score
                particle.person_best = particle.position

            if fintess_score > self.global_fitness_score:
                self.global_fitness_score = fintess_score
                self.global_best = particle.position


class MCEigen:
    def __init__(self, data, config: PSOConfig):
        self.config = config
        self.n_particles = config.n_particles
        self.amplitude = config.amplitude
        self.n_components = config.n_components
        self.rank = config.rank
        self.init_scale = config.init_scale
        self.T2 = config.T2
        self.T1 = config.T1
        self.data = data

        r_1 = 0.6
        r_2 = 0.8
        r_1_w = 0.42
        r_2_w = 0.57
            
        self.r_1 = {
            'weights' : r_1_w,
            'means' : r_1,
            'eigenvalues_prec' : r_1,
            'givens_angles' : r_1
        }
        
        self.r_2 = {
            'weights' : r_2_w,
            'means' : r_2,
            'eigenvalues_prec' : r_2,
            'givens_angles' : r_2
        }
        self.log_file = 'log_eigen.txt'

    def basic_gmm_init(self):
        
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', n_init=self.n_particles, max_iter=self.T2 * self.T1,  init_params='k-means++')
        gmm.fit(self.data)
        weights = gmm.weights_
        means = gmm.means_
        
        basic_prec_matr = np.zeros_like(gmm.precisions_cholesky_)
        for i in range(gmm.precisions_cholesky_.shape[0]):
            basic_prec_matr[i] = gmm.precisions_cholesky_[i] @ gmm.precisions_cholesky_[i].T

        self.basic_gmm_score = gmm.score(self.data)

        return weights, means, basic_prec_matr, gmm

    def update_global_best(self):
        pass

    def run(self):
        with open(self.log_file, 'w+') as f:
            max_iter = 31
            T1 = 10
            # make inital GMM
            # scatter points around init GMM
            # contol amplitude of scattering by eigenvalues max values
            # choose global best
            # reorder w.r.t. global best
            start = time.time()
            weights, means, basic_prec_matr, gmm = self.basic_gmm_init()
            print("Time for GMM init: ", time.time() - start, ' sec')
            basic_em_score = gmm.score(self.data)
            f.write('Basic GMM LL: ' + str(gmm.score(self.data)) + '\n')

            
            eigvals = [np.mean(np.linalg.eigvals(basic_prec_matr[i])) for i in range(basic_prec_matr.shape[0])]

            coef = self.config.eigvals_coef
            
            self.config.eig_val_max  = [eigvals[i] * coef for i in range(len(eigvals))]

            particles = [EigenParticle(self.n_components, self.data[0].shape[0], self.amplitude, weights, means, basic_prec_matr, self.data, eig_val_max=self.config.eig_val_max) for i in range(self.n_particles)]

            f.flush()
            best_pso_em_score = -np.inf

            for i in range(max_iter):
                if (i % T1) == 0:
                    f.write(f'Iter {i}\n')
                    start = time.time()
                    best_particle = particles[0]
                    for j in range(len(particles)):
                        new_ll = particles[j].run_em(self.data)
                        f.write('New LL: ' + str(new_ll) + '\n')
                        if best_pso_em_score < new_ll:
                            best_pso_em_score = new_ll
                            best_particle = particles[j]
                    print("Time for GMM reinit: ", time.time() - start, ' sec')
                    
                    start = time.time()
                    for i in range(len(particles)):
                        f.write(f'Particle LL: {particles[i].calculate_LL(self.data)} \n')
                    print("Time for LL calculation (including QRGivens): ", time.time() - start, ' sec')
                    f.flush()
                    means = best_particle.position['means']
                    weights = best_particle.position['weights']
                    basic_prec_matr = best_particle.get_cov_matrices()
                    particles = [EigenParticle(self.n_components, self.data[0].shape[0], self.amplitude, weights, means, basic_prec_matr, self.data, eig_val_max=self.config.eig_val_max) for i in range(self.n_particles)]
                    
                    # init GMM from PSO particle coordinates
                    # collect new LL 
                    # DO NOT reconstruct particles from new GMM initializations
                    pass

                c_1 = np.random.uniform(0, 1)
                c_2 = np.random.uniform(0, 1)
                for i in range(len(particles)):
                    particles[i].step(c_1, c_2, self.r_1, self.r_2)
                f.flush()
            
            # update personal best and global best

            # run reference EM

            # 2 times more iterations
            self.T2 = self.T2 * 2
            weights, means, basic_prec_matr, ref_gmm = self.basic_gmm_init()
            ref_em_score = ref_gmm.score(self.data)

        return {'em': basic_em_score, 'pso': best_pso_em_score, 'ref_em': ref_em_score}


class PSOEigen:
    def __init__(self, data, config: PSOConfig):
        self.config = config
        self.n_particles = config.n_particles
        self.amplitude = config.amplitude
        self.amplitude = 0.001

        self.n_components = config.n_components
        self.rank = config.rank
        self.init_scale = config.init_scale
        self.T2 = config.T2
        self.T1 = config.T1
        self.data = data

        r_1 = 0.6
        r_2 = 0.8
        r_1_w = 0.42
        r_2_w = 0.57
            
        self.r_1 = {
            'weights' : r_1_w,
            'means' : r_1,
            'eigenvalues_prec' : r_1,
            'givens_angles' : r_1
        }
        
        self.r_2 = {
            'weights' : r_2_w,
            'means' : r_2,
            'eigenvalues_prec' : r_2,
            'givens_angles' : r_2
        }
        self.log_file = 'log_eigen.txt'
        self.global_fitness_score = -np.inf
        

    def basic_gmm_init(self):
        
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', n_init=self.n_particles, max_iter=self.T2 * self.T1,  init_params='k-means++')
        gmm.fit(self.data)
        weights = gmm.weights_
        means = gmm.means_
        
        basic_prec_matr = np.zeros_like(gmm.precisions_cholesky_)
        for i in range(gmm.precisions_cholesky_.shape[0]):
            basic_prec_matr[i] = gmm.precisions_cholesky_[i] @ gmm.precisions_cholesky_[i].T

        self.basic_gmm_score = gmm.score(self.data)


        return weights, means, basic_prec_matr, gmm

    def init_global_best(self):
        
        self.global_best = self.particles[0]

        global_best_score = self.global_best.calculate_LL(self.data)

        for p in self.particles:
            score = p.calculate_LL(self.data)
            if score > global_best_score:
                    global_best_score = score
                    self.global_best = p

        for p in self.particles:
            p.reorder_wrt(self.global_best.position)

    def run(self):
        with open(self.log_file, 'w+') as f:
            # make inital GMM
            # scatter points around init GMM
            # contol amplitude of scattering by eigenvalues max values
            # choose global best
            # reorder w.r.t. global best
            start = time.time()
            weights, means, basic_prec_matr, gmm = self.basic_gmm_init()
            print("Time for GMM init: ", time.time() - start, ' sec')
            basic_em_score = gmm.score(self.data)
            f.write('Basic GMM LL: ' + str(gmm.score(self.data)) + '\n')
            
            eigvals = [np.mean(np.linalg.eigvals(basic_prec_matr[i])) for i in range(basic_prec_matr.shape[0])]

            coef = self.config.eigvals_coef
            
            self.config.eig_val_max  = [eigvals[i] * coef for i in range(len(eigvals))]

            self.particles = [EigenParticle(self.n_components, self.data[0].shape[0], self.amplitude, weights, means, basic_prec_matr, self.data, means_coef=self.config.eigvals_coef, eig_val_max=self.config.eig_val_max) for i in range(self.n_particles)]
            self.particles[-1] = EigenParticle(self.n_components, self.data[0].shape[0], self.amplitude, weights, means, basic_prec_matr, self.data, means_coef=0, eig_val_max=0)
            f.flush()
            self.init_global_best()

            best_pso_em_score = -np.inf
            best_ll_after_transforn_score = -np.inf

            best_particle = self.particles[0]

            for i in range(self.T1):
                f.write(f'Iter {i}\n')
                start = time.time()
                for j in range(len(self.particles)):
                    new_ll = self.particles[j].run_em(self.data, self.T2)

                    if best_pso_em_score < new_ll:
                        best_pso_em_score = new_ll
                        best_particle = self.particles[j]
                    f.write('New LL: ' + str(new_ll) + '\n')
                print("Time for GMM reinit: ", time.time() - start, ' sec')
                
                for j in range(len(self.particles)):
                    new_ll = self.particles[j].calculate_LL(self.data)


                    # if best_ll_after_transforn_score < score:
                    #     best_ll_after_transforn_score = score

                    f.write(f'Particle LL: {self.particles[j].calculate_LL(self.data)} \n')
                # print("Time for LL calculation (including QRGivens): ", time.time() - start, ' sec')
                # f.flush()

                # means = best_particle.position['means']
                # weights = best_particle.position['weights']
                # basic_prec_matr = best_particle.get_cov_matrices()
                self.particles = [EigenParticle(self.n_components, self.data[0].shape[0], self.amplitude, weights, means, basic_prec_matr, data=self.data, means_coef=self.config.eigvals_coef, eig_val_max=self.config.eig_val_max) for i in range(self.n_particles)]
                self.init_global_best()

                # for i in range(len(self.particles)):
                #     score = self.particles[i].calculate_LL(self.data)
                #     if best_ll_after_transforn_score < score:
                #         best_ll_after_transforn_score = score

                #     f.write(f'Particle LL (regular step): {self.particles[i].calculate_LL(self.data)} \n')

                for i in range(20):
                    c_1 = np.random.uniform(0, 1)
                    c_2 = np.random.uniform(0, 1)
                    for i in range(len(self.particles)):
                        self.particles[i].step(c_1, c_2, self.r_1, self.r_2)
                    f.flush()

                    changed = False
                    for particle in self.particles:
                        if particle.person_best_fitness_score > self.global_fitness_score:
                            self.global_fitness_score = particle.person_best_fitness_score
                            self.global_best = particle
                            changed = True

                    for p in self.particles:
                        p.global_best = self.global_best.position
                    
                    if changed:
                        for particle in self.particles:
                            particle.reorder_wrt(self.global_best.position)

            # run reference EM

            # 2 times more iterations
            self.T2 = self.T2 * 2
            weights, means, basic_prec_matr, ref_gmm = self.basic_gmm_init()
            ref_em_score = ref_gmm.score(self.data)

            f.write('Personal bests:')
            for particle in self.particles:
                f.write(str(particle.person_best_fitness_score))
            f.write(f'\n Global best: {self.global_fitness_score}')

        return {'em': basic_em_score, 'pso': best_pso_em_score, 'transformed': best_ll_after_transforn_score, 'ref_em': ref_em_score}

