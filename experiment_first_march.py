### add argument alpha - list of values
### add argement T1 - list of values
### add argement dataset 
### add argement algo_version - restart/no_restart
from argparse import ArgumentParser,BooleanOptionalAction
from utils import load_cloud_dataset, load_breast_cancer, load_seg_data, load_digits_dataset, load_satelite_dataset, load_synthetic_dataset
from pso import PSOConfig
from pso_v2.pso import PSOEigen
import os
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

def load_dataset(dataset_name):
    if dataset_name == 'seg':
        return load_seg_data()
    elif dataset_name == 'breast_cancer':
        return load_breast_cancer()
    elif dataset_name == 'landsat':
        return load_satelite_dataset()
    elif os.path.exists(dataset_name):
        return load_synthetic_dataset(dataset_name)
        
    raise ValueError('Wrong dataset name')

if __name__ == "__main__":
    parser = ArgumentParser(
                    prog = 'Experiment')
    parser.add_argument('-a', nargs='+', type=float, help='Alphas for experiment')
    parser.add_argument('--dataset', type=str, help='Dataset name or path to dataset')
    parser.add_argument('--particle_reinit', action=BooleanOptionalAction, help='Are particles being reinit each EM + PSO round')
    parser.add_argument('--n_runs', type=int, help='Number of runs')
    parser.add_argument('--config', type=str, help='PAth to config for PSO')
    parser.add_argument('-N', type=int, help='n particles')
    parser.add_argument('-M', type=int, help='Number of PSO + EM iterations')
    parser.add_argument('--n_comp', type=int, help='Number of components')

    args = parser.parse_args()
    print(args)
    data = load_dataset(args.dataset)
    config = PSOConfig.from_json(os.path.join('configs', 'default_params.json'))

    results = pd.DataFrame(columns=['T1', 'LogLikelihood, mean', 'LogLikelihood, std', 'Alpha', 'Dataset'])

    results_matrix = np.zeros([args.n_runs, len(args.a)])

    args.T2 = 100
    args.T1 = args.M
    config.n_particles = args.N
    config.n_components = args.n_comp

    gmms_start = [GaussianMixture(n_components=config.n_components, covariance_type='full', n_init=config.n_particles, max_iter=args.T2 * args.T1,  init_params='k-means++', verbose=False, verbose_interval=1) for i in range(args.n_runs)]
    for gmm in gmms_start:
        gmm.fit(data)

    print(f'Initial GMM res: {[gmm.score(data)for gmm in gmms_start]}\nMean: {np.mean([gmm.score(data)for gmm in gmms_start]):.3f} +- {np.std([gmm.score(data)for gmm in gmms_start]):.3f}')

    for j, alpha in enumerate(args.a):
        print(f'Alpha: {alpha}')
        for i in tqdm(list(range(args.n_runs))):
            config.eigvals_coef = alpha
            config.T1 = args.T1
            config.particle_reinit = args.particle_reinit
            
            pso_algo = PSOEigen(data, config, verbose=False, gmm_start=gmms_start[i])
            
            results_matrix[i, j] = pso_algo.run()['pso']
        res_dict = {'T1': args.T1, 'Alpha': alpha, 'Dataset': args.dataset, 'LogLikelihood, mean': np.array(results_matrix[:, j]).mean(), 'LogLikelihood, std': np.array(results_matrix[:, j]).std()}
        print(res_dict)
        results = results.append(res_dict, ignore_index=True)

    res_dict_em = {'T1': args.T1, 'Alpha': None, 'Dataset': args.dataset, 'LogLikelihood, mean': np.array([gmm.score(data)for gmm in gmms_start]).mean(), 'LogLikelihood, std': np.array([gmm.score(data)for gmm in gmms_start]).std()}
    results = results.append(res_dict_em, ignore_index=True)

    print('Results: ')
    print(results)
    now = datetime.datetime.now()
    now_str = now.strftime("%d_%m_%Y_%H_%M_%S")
    if not os.path.exists('exp_results'):
        os.mkdir('exp_results')
    results_save_name = f'Experiment_{now_str}'
    results.to_csv(os.path.join('exp_results', results_save_name))
    