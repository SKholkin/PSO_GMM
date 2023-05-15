import numpy as np
from utils import eigh_with_fixed_direction_range
from python_example import QRGivens_double as QRGivens


class GMMState:
    def __init__(self, weights, means, spd_matrices):
        self.weights = weights
        self.means = means
        self.spd_matrices = spd_matrices
        self.n_comp = self.weights.shape[0]
        self.dim = self.means.shape[1]

    @staticmethod
    def from_gmm(gmm):
        weights = gmm.weights_
        means = gmm.means_
        spd_matrices = gmm.precisions_
        return GMMState(weights, means, spd_matrices)

    def to_eigen(self):
        eigenvalues, givens_angles = np.zeros([self.n_comp, self.dim]), np.zeros([self.n_comp, int(self.dim * (self.dim - 1) / 2 )])

        for i in range(self.n_comp):
            eivals, v = eigh_with_fixed_direction_range(self.spd_matrices[i])
            givens_rotations = QRGivens(v)
            
            eigenvalues[i] = eivals
            givens_angles[i] = np.squeeze(givens_rotations)

        return {
            'weights' : self.weights,
            'means' : self.means,
            'eigenvalues' : eigenvalues,
            'givens_angles' : givens_angles
            }
    
    def to_normal(self):
        return {
            'weights' : self.weights,
            'means' : self.means,
            'spd': self.spd_matrices
            }

class Particle:
    def __init__(self):
        self.state = None
        self.pb = None
        self.pb_score = -np.inf
        self.gb = None
    
    def pso_step(self):
        pass
    
    def check_pb(self):
        if self.score() > self.pb_score:
            self.pb = self.copy()
    
    def score(self):
        pass

    def opt_step(self):
        pass
    
    def set_gb(self, gb):
        self.gb = gb.copy()
        
    def mutation(self):
        raise NotImplementedError('Mutation is not implemented')
    
    def get_best_score(self):
        return self.gb_score
        
class PSO:
    def __init__(self, particles, n_steps):
        self.particles = particles
        self.n_steps = n_steps
        self.gb_position, self.gb_score = None, -np.inf
        self.since_last_gb_change = 0
        self.check_gb()

        self.init_em_best = self.gb_score
        
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
            
    def check_for_mutation(self):
        scores = []
        for i, particle in enumerate(self.particles):
            p_score = particle.score()
            scores.append(p_score)
            if np.absolute(p_score - self.gb_score) < 5e-2:
                particle.mutation()
        
    def run(self):
        print(f'Init: gb ll {self.gb_score}')
        step = 0
        while True:
        # for step in range(self.n_steps):
            for i, particle in enumerate(self.particles):
                particle.pso_step()
                particle.opt_step()
                particle.check_pb()
            self.check_gb()
            # self.check_for_mutation()
            step += 1
            if self.since_last_gb_change >= self.n_steps:
                break
        
            print(f'Step {step}: gb ll {self.gb_score}')
        return self.gb_score
    