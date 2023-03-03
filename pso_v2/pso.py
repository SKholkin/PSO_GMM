import numpy as np
from sklearn.mixture import GaussianMixture
from pathlib import Path
import json
from copy import deepcopy
from addict import Dict
import datetime
from pso_v2.particle import EigenParticle
from utils import find_closest_spd
import time
from utils import Givens2Matrix, QRGivens, eigh_with_fixed_direction_range, find_closest_spd
from python_example import Givens2Matrix_double as Givens2Matrix
from python_example import QRGivens_double as QRGivens
import random

from gmm import GMM, precisions_to_sigmas_and_vice_versa

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
        

class PSOEigen:
    def __init__(self, data, config: PSOConfig, verbose=False, gmm_start=None):
        self.config = config
        self.n_particles = config.n_particles
        self.amplitude = config.amplitude
        self.amplitude = 0.001
        self.gmm_start = gmm_start

        self.verbose = verbose
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
        self.particle_trajectories = np.zeros((self.n_particles, self.T1))

    def basic_gmm_init(self, gmm_start=None):
        if gmm_start is not None:
            gmm = gmm_start
        else:
            gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', n_init=self.n_particles, max_iter=self.T2 * self.T1,  init_params='k-means++', verbose=self.verbose, verbose_interval=1)
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

        for i, p in enumerate(self.particles):
            score = p.calculate_LL(self.data)
            if score > global_best_score:
                    global_best_score = score
                    self.global_best = p

                    self.global_best_id = i

        for p in self.particles:
            p.reorder_wrt(self.global_best.position)

    def check_diversity(self):
        self.global_best
        for i, p in enumerate(self.particles):
            # if random.randrange(10) == 0:
            #     print('Lottery ticket!: ', i)
            #     p.mutate()

            if not p.diversity(self.global_best) and i != self.global_best_id:
                if self.verbose:
                    print('MUTATION!')
                
                p.mutate()


    def run(self):
        with open(self.log_file, 'w+') as f:
            start = time.time()

            weights, means, basic_prec_matr, gmm = self.basic_gmm_init(self.gmm_start)

            if self.verbose:
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

            best_particle = self.particles[0]

            for i in range(self.T1):
                f.write(f'Iter {i}\n')
                start = time.time()

                for j in range(len(self.particles)):
                    new_ll = self.particles[j].calculate_LL(self.data)
                    f.write(f'Particle LL Before EM: {self.particles[j].calculate_LL(self.data)} \n')
                for j in range(len(self.particles)):
                    ## replace 
                    # basic_prec_matr = self.particles[j].basic_prec_matr
                    # prec_matrcies = np.zeros_like(basic_prec_matr)
                    # # construct prec matr addition
                    # for i in range(self.n_components):
                    #     v = Givens2Matrix(np.expand_dims(self.particles[j].position['givens_angles'][i], axis=1))
                    #     addition = v @ np.diag(self.particles[j].position['eigenvalues_prec'][i]) @ v.T
                    #     prec_matrcies[i] = find_closest_spd(basic_prec_matr[i] + addition)

                    # sigmas = precisions_to_sigmas_and_vice_versa(prec_matrcies)
                    # gmm = GMM(weights.shape[0], dim=self.data.shape[1], init_mu=self.particles[j].position['means']
                    #           ,init_sigma=sigmas, init_pi=self.particles[j].position['weights'])

                    # gmm.init_em(self.data)

                    # new_ll, final_iter = gmm.run_em(self.T2)
                    # print('Final iter for scattered particle: ', final_iter, 'LL: ', new_ll)

                    # new_weights = gmm.pi
                    # new_means = gmm.mu
                    # new_precisions = precisions_to_sigmas_and_vice_versa(gmm.sigma)

                    # self.particles[j].set_params_after_em(new_weights, new_means, new_precisions, new_ll)
                    
                    new_ll = self.particles[j].run_em(self.data, self.T2)
                    self.particle_trajectories[j, i] = new_ll
                    if self.verbose:
                        print(f'Particle {j} New LL', new_ll)

                    if best_pso_em_score < new_ll:
                        best_pso_em_score = new_ll
                        best_particle = self.particles[j]
                    # f.write('New LL: ' + str(new_ll) + '\n')
                if self.verbose:
                    print("Time for GMM reinit: ", time.time() - start, ' sec')
                
                # for j in range(len(self.particles)):
                #     new_ll = self.particles[j].calculate_LL(self.data)
                #     f.write(f'Particle LL: {self.particles[j].calculate_LL(self.data)} \n')

                self.check_diversity()

                # means = best_particle.position['means']
                # weights = best_particle.position['weights']
                # basic_prec_matr = best_particle.get_cov_matrices()
                if self.config.particle_reinit:
                    if self.verbose:
                        print('Paricles reinit')
                    self.particles = [EigenParticle(self.n_components, self.data[0].shape[0], self.amplitude, weights, means, basic_prec_matr, data=self.data, means_coef=self.config.eigvals_coef, eig_val_max=self.config.eig_val_max) for i in range(self.n_particles)]
                    self.init_global_best()

                # for i in range(len(self.particles)):
                #     score = self.particles[i].calculate_LL(self.data)
                #     if best_ll_after_transforn_score < score:
                #         best_ll_after_transforn_score = score

                #     f.write(f'Particle LL (regular step): {self.particles[i].calculate_LL(self.data)} \n')

                n_pso_updates = 30
                for i in range(n_pso_updates):
                    c_1 = np.random.uniform(0, 1)
                    c_2 = np.random.uniform(0, 1)
                    for i in range(len(self.particles)):
                        self.particles[i].step(c_1, c_2, self.r_1, self.r_2)
                    f.flush()

                    changed = False
                    for i, particle in enumerate(self.particles):
                        if particle.person_best_fitness_score > self.global_fitness_score:
                            self.global_fitness_score = particle.person_best_fitness_score
                            self.global_best = particle
                            self.global_best_id = i
                            changed = True

                    for p in self.particles:
                        p.global_best = self.global_best.position
                    
                    if changed:
                        for particle in self.particles:
                            particle.reorder_wrt(self.global_best.position)

            # run reference EM

            f.write('Personal bests:')
            for particle in self.particles:
                f.write(str(particle.person_best_fitness_score))
            f.write(f'\n Global best: {self.global_fitness_score}')

        return {'em': basic_em_score, 'pso': best_pso_em_score}

