import numpy as np
from sklearn.mixture import GaussianMixture
from pathlib import Path
import json
from copy import deepcopy
from addict import Dict
import datetime


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
        self.keys = ['weights', 'delta_mean', 'delta_diag_prec', 'delta_param_prec']
        
        self.velocity = { 
            'weights' : np.zeros(n_components),
            'delta_mean' : np.zeros((n_components, data_dim)),
            'delta_diag_prec' : np.zeros((n_components, data_dim)),
            'delta_param_prec' : np.zeros((n_components, rank, data_dim))
        }

        self.position = {
            'weights' : weights,
            'delta_mean' : np.random.normal(0, init_scale, size=(n_components, data_dim)),
            'delta_diag_prec' : np.random.normal(0, init_scale, size=(n_components, data_dim)),
            'delta_param_prec' : np.random.normal(0, init_scale, size=(n_components, rank, data_dim))
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
            'delta_param_prec' : r_1
        }
        
        self.r_2 = {
            'weights' : r_2_w,
            'delta_mean' : r_2,
            'delta_diag_prec' : r_2,
            'delta_param_prec' : r_2
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

        self.particles = [Particle(self.n_components, self.data_dim, self.rank, self.amplitude, self.init_scale, self.basic_weights)
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

        means = deepcopy(self.basic_means) + particle.position['delta_mean']

        prec_matr = deepcopy(self.basic_prec_matr)

        cholesky = np.zeros_like(prec_matr)
        
        for i in range(self.n_components):

            prec_matr[i] += np.diag(particle.position['delta_diag_prec'][i] ** 2)

            for k in range(self.config.rank):
                prec_matr[i] += particle.position['delta_param_prec'][i][k] @ particle.position['delta_param_prec'][i][k].T

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
        
    def reinit_particles(self, inplace=True):
        particles = [Particle(self.n_components, self.data_dim, self.rank, self.amplitude, self.init_scale, self.basic_weights) for _ in range(self.n_particles)]
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

