from pso import PSO, PSOConfig
from addict import Dict
from copy import deepcopy
import csv
import datetime
import numpy as np
import argparse
from utils import load_dataset, real_dataset_names
from sklearn.mixture import GaussianMixture
from time import time

def pso_vs_em_experiment(config, data):
    pso = PSO(data, config)
    n_iters = config.n_iters
    T1 = config.T1
    
    time_pso = 0
    time_big_em = 0

    gmm_reinit_pso_scores = []
    print('Basic EM score', pso.basic_gmm_score)
    start_pso = time()
    for i in range(n_iters):
        pso.step()
        if  i % (n_iters // T1) == 0:
            print(f'Iter {i} Particles reinit')

            rerun_em_fintess_score_list = pso.run_em_particles_fintess_score()
            fintess_score_list = pso.get_particles_fitness_scores()

            gmm_reinit_pso_scores.append(max(rerun_em_fintess_score_list))

            print('PSO: Rerun EM best fintess score', max(rerun_em_fintess_score_list))
            print(f'PSO: Max fitness score {max(fintess_score_list)}')
    init_gmm_em_score = pso.basic_gmm_score

    time_pso = time() - start_pso

    print('PSO time, sec', time_pso)

    # init big EM
    
    T1 = config.T1
    T2 = config.T2

    start_big_em = time()
    
    reference_gmm = GaussianMixture(n_components=config.n_components, covariance_type='full', n_init=10 * config.n_particles, max_iter=T2 * T1,  init_params=config.EM_init_method)
    reference_gmm.fit(data)

    time_big_em = time() - start_big_em
    
    print('Big EM time, sec', time_big_em)

    print('Reference GMM 2 * n_particles inits, T1 * T2 iters:', reference_gmm.score(data))

    return {'em': init_gmm_em_score, 'pso': max(gmm_reinit_pso_scores), 'pso_time': time_pso, 'ref_gmm_score': reference_gmm.score(data), 'time_big_em': time_big_em}
    

def save_to_csv(config: Dict, results: Dict):
    new_dict = deepcopy(config)
    new_dict.update(results)
    
    now = datetime.datetime.now()
    now_str = now.strftime("%d_%m_%Y_%H_%M_%S")

    with open(f'logs/experiment_{new_dict.dataset}_{now_str}_by_time.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = new_dict.keys())
        writer.writeheader()
        writer.writerow(new_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help=f'Config of PSO')
    parser.add_argument('--dataset', type=str, help=f'Real dataset name {real_dataset_names} or path to synthetic dataset')
    parser.add_argument('--n_runs', type=int, help=f'Number of runs')
    args = parser.parse_args()

    config = PSOConfig.from_json(args.config)

    config.update({'dataset': args.dataset})
    data = load_dataset(args.dataset)

    n_runs = args.n_runs
    results = {}
    results_list = []
    for run in range(n_runs):
        results_list.append(pso_vs_em_experiment(config, data))

    for key in results_list[0].keys():
        results[key] = f'{np.mean([item[key] for item in results_list])} +- {np.std([item[key] for item in results_list])}'

    save_to_csv(config, results)
    