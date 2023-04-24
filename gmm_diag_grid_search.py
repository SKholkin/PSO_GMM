from utils import load_dataset

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score, silhouette_score
from sklearn.cluster import KMeans

from copy import deepcopy

from utils import save_results_string
import itertools

from gmm_diag import PSO, Particle, GMMDiagonal

def create_diag_particles(n_particles, n_comp, data, inertia=0.4, r_1=0.4, r_2=0.6):
    particles = []
    for i in range(n_particles):
        gmm = GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=100, n_init=1)
        gmm.fit(data)
        
        weights = gmm.weights_
        means = gmm.means_
        var = 1 / gmm.precisions_
        state = {'weights': gmm.weights_, 'means': gmm.means_, 'var': gmm.covariances_}
        particles.append(state)
        
    return [GMMDiagonal(data, n_comp, state, inertia, r_1, r_2) for state in particles]

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

    inertia_list, r_1_list, r_2_list = [0.2, 0.4, 0.6, 0.8, 1], [0.2, 0.4, 0.6, 0.8, 1], [0.2, 0.4, 0.6, 0.8, 1]
    inertia_list, r_1_list, r_2_list = [0.5, 1], [0.5], [0.5, 1]

    desc = f'{dataset_name}_{n_comp}_comp_N_{N}_M_{M}'

    result_string = ''
    for inertia, r_1, r_2 in itertools.product(inertia_list, r_1_list, r_2_list):

        scores = []
        for n in range(n_runs):
            particles = create_diag_particles(N, n_comp, data, inertia, r_1, r_2)
            pso = PSO(particles, M)
            best_score = pso.run()
            scores.append(best_score)
            print(best_score)

        result_string += f'{inertia, r_1, r_2}:  PSO + EM: {np.mean(scores)} +- {np.std(scores)}\n'

    scores = []
    for n in range(n_runs):
        gmm = GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=100, n_init=N)
        gmm.fit(data)
        scores.append(gmm.score(data))

    result_string += f'EM: {np.mean(scores)} +- {np.std(scores)}\n'

    print(result_string)

    save_results_string(result_string, desc + '_Grid_Search')
