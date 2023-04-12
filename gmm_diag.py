from utils import load_dataset

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score, silhouette_score
from sklearn.cluster import KMeans


from copy import deepcopy

class Particle:
    def __init__(self):
        self.state = None
        self.pb = None
        self.pb_score = -np.inf
        self.gb = None
    
    def step(self):
        pass
    
    def check_pb(self):
        if self.score() > self.pb_score:
            self.pb = self.copy()
    
    def score(self):
        pass

    def finetune(self):
        pass
    
    def set_gb(self, gb):
        self.gb = gb.copy()
        
    def mutation(self):
        raise NotImplementedError('Mutation is not implemented')
        
    
class PSO:
    def __init__(self, particles, n_steps):
        self.particles = particles
        self.n_steps = n_steps
        self.gb_particle, self.gb_score = None, -np.inf
        self.check_gb()
        
    def check_gb(self):
        for particle in self.particles:
            score = particle.score()
            if score > self.gb_score:
                self.gb_score = score
                self.gb_particle = particle.copy()
                
        gb = self.gb_particle.copy()
        for particle in self.particles:
            particle.set_gb(gb)
            
    def check_for_mutation(self):
        self.gb_score
        for i, particle in enumerate(self.particles):
            p_score = particle.score()
            if np.absolute(p_score - self.gb_score) < 5e-2:
                particle.mutation()
        
    def run(self):
        print(f'Init: gb ll {self.gb_score}')
        for step in range(self.n_steps):
            for i, particle in enumerate(self.particles):
                particle.step()
                particle.finetune()
                particle.check_pb()
#                 print(f'{i} score {particle.score()}')
#             print(f'Step {step} {[item.score() for item in self.particles]}')
            self.check_gb()
#             print(f'Step {step}: gb ll {self.gb_score}')
            self.check_for_mutation()
        return self.gb_score
        
        
    
class GMMDiagonal(Particle):
    def __init__(self, data, n_comp, init_state, intertia, r_1, r_2):
        super(Particle, self).__init__()
        self.state = init_state
        self.intertia, self.r_1, self.r_2 = intertia, r_1, r_2
        self.velocity = {key: np.zeros_like(value) for key, value in init_state.items()}
        self.gb = None
        self.data = data
        self.n_comp = n_comp
        self.pb, self.pb_score = None, -np.inf
        self.history = []
        
    def step(self):
        if self.pb is None:
            self.pb = self.copy()
        c_1, c_2 = np.random.uniform(), np.random.uniform()
        for key in self.velocity.keys():
            self.velocity[key] = self.intertia * self.velocity[key] + self.r_1 * c_1 * (self.pb.state[key] - self.state[key]) + self.r_2 * c_2 * (self.gb.state[key] - self.state[key])
#             self.velocity[key] += 0.001 * np.random.normal(size=self.velocity[key].shape)
            self.state[key] += self.velocity[key]
        self.state['var'] = np.absolute(self.state['var'])
        self.state['weights'] = np.absolute(self.state['weights'])
        self.state['weights'] = self.state['weights'] / np.sum(self.state['weights'])
        
    def mutation(self):
        self.state['means'] += 0.01 * np.random.normal(size=self.state['means'].shape)
#         print(np.mean(self.state['means']))
        self.state['var'] += 0.1 * np.random.uniform() * self.state['var']
        
    def score(self):
        # create sklearn.mixture.Gmm from self.state
        
        means = self.state['means']
        var = self.state['var']
        weights = self.state['weights']
        
        gmm = GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=0)
        
        cholesky = 1 / np.sqrt(var)
        gmm.weights_ = weights
        gmm.means_ = means
        gmm.precisions_cholesky_ = cholesky
        
        return gmm.score(self.data)
    
    def finetune(self):
        
        means = self.state['means']
        var = self.state['var']
        prec = 1 / var
        weights = self.state['weights']
        
        gmm = GaussianMixture(n_components=self.n_comp, covariance_type='diag', weights_init=weights, means_init=means, precisions_init=prec, max_iter=100,  verbose=0, verbose_interval=1)
        gmm.fit(self.data)
        
        weights = gmm.weights_
        means = gmm.means_
        var = 1 / gmm.precisions_
        
        self.state['means'] = means
        self.state['var'] = var
        self.state['weights'] = weights
        
        self.history.append(self.score())

    def copy(self):
        state_copy = deepcopy(self.state)
        return GMMDiagonal(self.data, self.n_comp, state_copy, self.intertia, self.r_1, self.r_2)
        
def create_diag_particles(n_particles, n_comp, data):
    particles = []
    for i in range(n_particles):
        gmm = GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=100)
        gmm.fit(data)
        
        weights = gmm.weights_
        means = gmm.means_
        var = 1 / gmm.precisions_
        state = {'weights': gmm.weights_, 'means': gmm.means_, 'var': gmm.covariances_}
        particles.append(state)
        
    return [GMMDiagonal(data, n_comp, state, 0.4, 0.4, 0.6) for state in particles]

n_runs = 1
scores = []
data = load_dataset('breast_cancer')
for n in range(n_runs):
    n_comp = 10
    particles = create_diag_particles(50, n_comp, data)
    pso = PSO(particles, 10)
    best_score = pso.run()
    scores.append(best_score)
    print(best_score)

gmm = GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=100, n_init=500)
gmm.fit(data)
print(f'{np.mean(scores)} +- {np.std(scores)} VS\n{gmm.score(data)}')
