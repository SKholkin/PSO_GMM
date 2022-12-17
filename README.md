## PSO_GMM
Repository for project about applying Particle Swarm Optimization to optimizing Gaussian Mixture Model parameters.


### Environment configuration
To install requirements, run
```bash
pip install requirements.txt
```

### Data generation
Real datasets are located in `/data`, synthetic can be generated with `synthetic_data_gen.py` in `.npz` format.

### Experiments
One can reproduce experiments by running 

`pso_vs_em_by_iters.py --config path_to_config --dataset *name of real dataset or path to synthetic dataset file* --n_runs *number of runs you want to average on*`

or 

`pso_vs_em_by_time.py --config path_to_config --dataset *name of real dataset or path to synthetic dataset file* --n_runs *number of runs you want to average on*`

the difference between those two is that second uses EM budget 10 * M * T_1 * T_2 because this approximately comparable with time PSO runs with EM budget 2 * M * T_1 * T_2 (remember PSO algo is not optimized yet!). 


### Optional 
run 
`pso_vs_scatter_vs_em_exp.py` if you want to also compare PSO with just randomly scattering points across manifold through our parametrization with no PSO update and running EM on these randomly initialized particles particles.
