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

Results are being saved as one row csv files to log folder

#### Configs

Some PSO configs are stored in `/configs` folder

### Examples

As a particular example you can run:

`python synthetic_data_gen.py -n_samples 1000 -dim 30 -c_sep 2 -n_comp 15`
`python pso_vs_em_by_iters.py --config configs/default_params_synth_30.json --n_runs 10 --dataset Synthetic_dim_30_n_samples_1000_n_comp_15_c_separation_2.0.data.npy`

or

`python pso_vs_em_by_iters.py --config configs/params_breast_cancer.json --n_runs 10 --dataset breast_cancer`

### Optional 
run 
`pso_vs_scatter_vs_em_exp.py` if you want to also compare PSO with just randomly scattering points across manifold through our parametrization with no PSO update and running EM on these randomly initialized particles particles.
