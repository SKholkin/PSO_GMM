from pso import PSO, PSOConfig
from addict import Dict
from copy import deepcopy
import csv
import datetime
import numpy as np
import argparse
from utils import load_dataset, real_dataset_names
from sklearn.mixture import GaussianMixture

def pso_vs_scatter_experiment(config, data):
    pso = PSO(data, config)
    scatter = PSO(data, config, basic_gmm=pso.basic_gmm) # not pso really
    n_iters = config.n_iters
    T1 = config.T1
    
    gmm_reinit_scatter_scores = []
    gmm_reinit_pso_scores = []
    print('Basic EM score', pso.basic_gmm_score)
    for i in range(n_iters):
        pso.step()
        if  i % (n_iters // T1) == 0:
            print(f'Iter {i} Particles reinit')
            scatter.reinit_particles()
            scatter_rerun_em_fintess_score_list = scatter.run_em_particles_fintess_score()
            gmm_reinit_scatter_scores.append(max(scatter_rerun_em_fintess_score_list))

            rerun_em_fintess_score_list = pso.run_em_particles_fintess_score()
            fintess_score_list = pso.get_particles_fitness_scores()

            gmm_reinit_pso_scores.append(max(rerun_em_fintess_score_list))

            print('PSO: Rerun EM best fintess score', max(rerun_em_fintess_score_list))
            print(f'PSO: Max fitness score {max(fintess_score_list)}')
            print('Scatter: Rerun EM best fintess score', max(scatter_rerun_em_fintess_score_list))
    init_gmm_em_score = pso.basic_gmm_score

    # init big EM
    
    T1 = config.T1
    T2 = config.T2
    
    reference_gmm = GaussianMixture(n_components=config.n_components, covariance_type='full', n_init=2 * config.n_particles, max_iter=T2 * T1,  init_params=config.EM_init_method)
    reference_gmm.fit(data)


    print('Reference GMM 2 * n_particles inits, T1 * T2 iters:', reference_gmm.score(data))

    return {'em': init_gmm_em_score, 'pso': max(gmm_reinit_pso_scores), 'scatter': max(gmm_reinit_scatter_scores), 'ref_gmm_score': reference_gmm.score(data)}
    

def save_to_csv(config: Dict, results: Dict):
    new_dict = deepcopy(config)
    new_dict.update(results)
    
    now = datetime.datetime.now()
    now_str = now.strftime("%d_%m_%Y_%H_%M_%S")

    with open(f'logs/experiment_{new_dict.dataset}_{now_str}.csv', 'w') as csvfile:
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
        results_list.append(pso_vs_scatter_experiment(config, data))

    for key in results_list[0].keys():
        results[key] = f'{np.mean([item[key] for item in results_list])} +- {np.std([item[key] for item in results_list])}'

    save_to_csv(config, results)

    print(f'EM: {np.mean([item["em"] for item in results_list])} PSO: {np.mean([item["pso"] for item in results_list])} Scatter: {np.mean([item["scatter"] for item in results_list])}')
    