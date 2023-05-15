
from pso_v3.eigen_pso import EigenParticle, GMMState, gmm_init_states, EigenPSO
from pso_v3.base import PSO
from sklearn.mixture import GaussianMixture
from argparse import ArgumentParser

from pso_v3.riemannian_pso import RiemannianPartilce
import os 
import numpy as np
from tqdm import tqdm
import pandas as pd

import datetime

def fn_1():
    from utils import load_seg_data
    data = load_seg_data()
    n_comp = 5
    gmm = GaussianMixture(n_comp, covariance_type='full')
    gmm.fit(data)
    print(f'Before: {gmm.score(data)}')

    init_state = GMMState.from_gmm(gmm)

    intertia, r_1, r_2 = 0.5, 0.5, 0.5

    particle = EigenParticle(data, n_comp, init_state, intertia, r_1, r_2)

    print(f'After: {particle.score()}')

    particle.opt_step()

    gmm_gb = GaussianMixture(n_comp, covariance_type='full')
    gmm_gb.fit(data)

    init_state_gb = GMMState.from_gmm(gmm)
    particle_gb = EigenParticle(data, n_comp, init_state_gb, intertia, r_1, r_2)

    particle.set_gb(particle_gb.position)

    particle.pso_step()
    particle.pso_step()

    print(f'After pso_step: {particle.score()}')

    particle.opt_step()

    print(f'After opt_step: {particle.score()}')
    particle.pso_step()
    
    print(f'After pso_step: {particle.score()}')

if __name__ == '__main__':

    from utils import load_seg_data, load_dataset
    parser = ArgumentParser(
                    prog = 'Experiment')
    parser.add_argument('-a', nargs='+', type=float, help='Alphas for experiment')
    parser.add_argument('--dataset', type=str, help='Dataset name or path to dataset')
    parser.add_argument('--n_runs', type=int, help='Number of runs')
    parser.add_argument('--config', type=str, help='PAth to config for PSO')
    parser.add_argument('-N', type=int, help='n particles')
    parser.add_argument('-M', type=int, help='Number of PSO + EM iterations')
    parser.add_argument('--n_comp', type=int, help='Number of components')


    args = parser.parse_args()
    print(args)
    data = load_dataset(args.dataset)
    
    n_comp = 5
    n_steps = 10
    n_particles = 20

    intertia, r_1, r_2 = 0.5, 0.5, 0.5

    now = datetime.datetime.now()
    now_str = now.strftime("%d_%m_%Y_%H_%M_%S")
    if not os.path.exists('exp_results'):
        os.mkdir('exp_results')
    results_save_name = f'Experiment_{args.dataset}_NComp_{args.n_comp}_N_{args.N}_M_{args.M}_{now_str}.csv'

    results = pd.DataFrame(columns=['T1', 'LogLikelihood', 'Dataset', 'PSO'])

    for n in tqdm(range(args.n_runs)):
        init_states = gmm_init_states(data, n_comp, n_particles)
        particles = [EigenParticle(data, n_comp, st, intertia, r_1, r_2) for st in init_states]
        
        eigen_pso = EigenPSO(particles, n_steps)
        eigen_pso.run()

        pso_res = eigen_pso.gb_score
        em_res = eigen_pso.init_em_best

        res_dict_pso = {'N': n_particles, 'M': args.M, 'Dataset': args.dataset, 'LogLikelihood': pso_res, 'PSO': True}
        res_dict_em = {'N': n_particles, 'M': None, 'Dataset': args.dataset, 'LogLikelihood': em_res, 'PSO': False}

        results = results._append(res_dict_pso, ignore_index=True)
        results = results._append(res_dict_em, ignore_index=True)

        results.to_csv(os.path.join('exp_results', results_save_name))


    # particles = [RiemannianPartilce(data, n_comp, st, intertia, r_1, r_2) for st in init_states]
    # riehmannian_pso = PSO(particles, n_steps)
    # riehmannian_pso.run()
