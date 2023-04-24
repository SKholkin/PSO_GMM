from utils import load_dataset

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score, silhouette_score
from sklearn.cluster import KMeans

from copy import deepcopy

from utils import save_results_string

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
        self.since_last_gb_change = 0
        self.check_gb()
        
    def check_gb(self):
        self.since_last_gb_change += 1
        for particle in self.particles:
            score = particle.score()
            if score > self.gb_score:
                self.gb_score = score
                self.gb_particle = particle.copy()
                self.since_last_gb_change = 0
            
        gb = self.gb_particle.copy()
        for particle in self.particles:
            particle.set_gb(gb)
            
    def check_for_mutation(self):
        self.gb_score
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
                particle.step()
                particle.finetune()
                particle.check_pb()
            self.check_gb()
            self.check_for_mutation()
            step += 1
            if self.since_last_gb_change >= self.n_steps:
                break
        
            print(f'Step {step}: gb ll {self.gb_score}')
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
        self.em_iters = 0
        
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
        self.state['var'] += 0.1 * np.random.uniform() * self.state['var']
        
    def score(self):
        
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
        self.em_iters += gmm.n_iter_
        
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
        gmm = GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=100, n_init=1)
        gmm.fit(data)
        
        weights = gmm.weights_
        means = gmm.means_
        var = 1 / gmm.precisions_
        state = {'weights': gmm.weights_, 'means': gmm.means_, 'var': gmm.covariances_}
        particles.append(state)
        
    return [GMMDiagonal(data, n_comp, state, 0.4, 0.4, 0.6) for state in particles]


from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(
                    prog = 'Experiment')
    parser.add_argument('-N', type=int, help='Number of particles')
    parser.add_argument('-M', type=int, help='Number of PSO steps for dynamic stopping')
    parser.add_argument('--dataset', type=str, help='Dataset name or path to dataset')
    parser.add_argument('--n_comp', type=int, help='Number of components (clusters)')
    parser.add_argument('--n_runs', type=int, help='Number of runs')

    args = parser.parse_args()
    print('Args:\n' + str(args))
    dataset_name = args.dataset
    n_runs = args.n_runs
    data = load_dataset(dataset_name)
    n_comp = args.n_comp
    M = args.M
    N = args.N

    desc = f'{dataset_name}_{n_comp}_comp_N_{N}_M_{M}'

    scores = []
    for n in range(n_runs):
        particles = create_diag_particles(N, n_comp, data)
        pso = PSO(particles, M)
        best_score = pso.run()
        scores.append(best_score)
        print(best_score)

    result_string = f'PSO + EM: {np.mean(scores)} +- {np.std(scores)}\n'

    scores = []
    for n in range(n_runs):
        gmm = GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=100, n_init=N)
        gmm.fit(data)
        scores.append(gmm.score(data))

    result_string += f'EM: {np.mean(scores)} +- {np.std(scores)}\n'

    print(result_string)

    save_results_string(result_string, desc)
