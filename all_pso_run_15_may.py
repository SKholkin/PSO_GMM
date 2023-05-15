from pso_v3.eigen_pso import EigenParticle, GMMState, gmm_init_states, EigenPSO
from pso_v3.low_rank_eigen import LowRankEigenParticle, LowRankEigenPSO, get_base_spd
from pso_v3.base import PSO
from pso_v3.scatter_init import gmm_scatter_around_best_init_states, get_best_gmm_state

from sklearn.mixture import GaussianMixture
from argparse import ArgumentParser

from pso_v3.riemannian_pso import RiemannianPartilce
import os 
import numpy as np
from tqdm import tqdm
import pandas as pd

import datetime
from utils import load_seg_data, load_digits_dataset

import pandas as pd

n_runs = 11

n_comp = 10
n_particles = 10
intertia, r_1, r_2 = 1, 1, 1
n_steps = 5

data = load_digits_dataset()
res_df = pd.DataFrame(columns=['Data', 'N_comp', 'N', 'M', 'Init', 'Param','LL', 'Base_score'])

for i in range(n_runs):

    init_states_eig_base, scores = gmm_init_states(data, n_comp, n_particles)

    best_state, best_score = get_best_gmm_state(init_states_eig_base, data, n_comp)

    print('Basic Eigen: ')
    
    particles = [EigenParticle(data, n_comp, st, intertia, r_1, r_2) for st in init_states_eig_base]

    eigen_pso = EigenPSO(particles, n_steps)
    eigen_pso.run()
    res_dict = {'Data': 'digits', 'N_comp': n_comp, 'N': n_particles, 'M': n_steps, 'Init': 'Points', 'Param': 'Eigen', 'LL': eigen_pso.gb_score, 'Base_score': best_score}

    res_df = res_df._append(res_dict, ignore_index=True)

    print('Delta Eigen:')

    base_spd = get_base_spd(init_states_eig_base)

    rank = data.shape[1]

    particles = [LowRankEigenParticle(data, n_comp, st, intertia, r_1, r_2, rank, base_spd) for st in init_states_eig_base]

    low_rank_eigen_pso = LowRankEigenPSO(particles, n_steps)

    low_rank_eigen_pso.run()

    res_dict = {'Data': 'digits', 'N_comp': n_comp, 'N': n_particles, 'M': n_steps, 'Init': 'Points', 'Param': 'DeltaEigen', 'LL': low_rank_eigen_pso.get_best_score(), 'Base_score': best_score}

    res_df = res_df._append(res_dict, ignore_index=True)

    print('Basic Eigen Scatter init:')

    init_states_scatter, base_spd = gmm_scatter_around_best_init_states(data, n_comp, n_particles, gmm_states=init_states_eig_base, std=0.3)

    particles = [EigenParticle(data, n_comp, st, intertia, r_1, r_2) for st in init_states_scatter]
    
    eigen_pso = EigenPSO(particles, n_steps)
    eigen_pso.run()

    res_dict = {'Data': 'digits', 'N_comp': n_comp, 'N': n_particles, 'M': n_steps, 'Init': 'Scatter', 'Param': 'Eigen', 'LL': eigen_pso.gb_score, 'Base_score': best_score}

    res_df = res_df._append(res_dict, ignore_index=True)

    print('Delta Eigen Scatter init:')
    
    particles = [LowRankEigenParticle(data, n_comp, st, intertia, r_1, r_2, rank, base_spd) for st in init_states_scatter]

    low_rank_eigen_pso = LowRankEigenPSO(particles, n_steps)
    low_rank_eigen_pso.run()

    res_dict = {'Data': 'digits', 'N_comp': n_comp, 'N': n_particles, 'M': n_steps, 'Init': 'Scatter', 'Param': 'DeltaEigen', 'LL': low_rank_eigen_pso.get_best_score(), 'Base_score': low_rank_eigen_pso.get_best_score()}

    res_df = res_df._append(res_dict, ignore_index=True)

    now = datetime.datetime.now()
    now_str = now.strftime("%d_%m_%Y_%H_%M_%S")

    results_save_name = f'Experiment_{"digits"}_N_Comp_{n_comp}_N_{n_particles}_M_{n_steps}_{now_str}.csv'

    res_df.to_csv(os.path.join('exp_results', results_save_name))
