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

def load_dataset(dataset_name):
    return load_seg_data()
    pass

if __name__ == "__main__":
    parser = ArgumentParser(
                    prog = 'Experiment')
    parser.add_argument('-a', nargs='+', type=float, help='Alphas for experiment')
    parser.add_argument('-T1', type=int, help='Number of EM + PSO rounds')
    parser.add_argument('--dataset', type=str, help='Dataset name or path to dataset')
    parser.add_argument('--particle_reinit', action=BooleanOptionalAction, help='Are particles being reinit each EM + PSO round')
    parser.add_argument('--n_runs', type=int, help='Number of runs')
    parser.add_argument('--config', type=str, help='PAth to config for PSO')

    args = parser.parse_args()
    print(args)
    data = load_dataset(args.dataset)
    config = PSOConfig.from_json(os.path.join('configs', 'default_params.json'))

    results = pd.DataFrame(columns=['T1', 'LogLikelihood, mean', 'LogLikelihood, std', 'Alpha', 'Dataset'])
    for alpha in args.a:
        stats = []
        for i in range(args.n_runs):
            config.eigvals_coef = alpha
            config.T1 = args.T1
            config.particle_reinit = args.particle_reinit
            
            pso_algo = PSOEigen(data, config)
            
            stats = pso_algo.run()['pso']

        print(f'Alpha = {alpha}, T1 = {args.T1}, Particels reinit {args.particle_reinit}: {np.array(stats).mean()} +- {np.array(stats).std()}')
        results = results.append({'T1': args.T1, 'Alpha': alpha, 'Dataset': args.dataset, 'LogLikelihood, mean': np.array(stats).mean(), 'LogLikelihood, std': np.array(stats).std()}, ignore_index=True)
    print(results)
    now = datetime.datetime.now()
    now_str = now.strftime("%d_%m_%Y_%H_%M_%S")
    if not os.path.exists('exp_results'):
        os.mkdir('exp_results')
    results_save_name = f'Experiment_{now_str}'
    results.to_csv(os.path.join('exp_results', results_save_name))
    