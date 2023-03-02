import numpy as np
from numpy import random
from scipy.stats import multivariate_normal


class GMM():
    def __init__(self, k, dim, init_mu=None, init_sigma=None, init_pi=None, colors=None):
        '''
        Define a model with known number of clusters and dimensions.
        input:
            - k: Number of Gaussian clusters
            - dim: Dimension 
            - init_mu: initial value of mean of clusters (k, dim)
                       (default) random from uniform[-10, 10]
            - init_sigma: initial value of covariance matrix of clusters (k, dim, dim)
                          (default) Identity matrix for each cluster
            - init_pi: initial value of cluster weights (k,)
                       (default) equal value to all cluster i.e. 1/k
            - colors: Color valu for plotting each cluster (k, 3)
                      (default) random from uniform[0, 1]
        '''
        self.k = k
        self.dim = dim
        if(init_mu is None):
            init_mu = random.rand(k, dim)*20 - 10
        self.mu = init_mu
        if(init_sigma is None):
            init_sigma = np.zeros((k, dim, dim))
            for i in range(k):
                init_sigma[i] = np.eye(dim)
        self.sigma = init_sigma
        if(init_pi is None):
            init_pi = np.ones(self.k)/self.k
        self.pi = init_pi
        if(colors is None):
            colors = random.rand(k, 3)
        self.colors = colors
        self.covariance_reg_value = 1e-06
        self.pi_reg = 1e-5
        self.threshold_frob = 1e-5
    
    def init_em(self, X):
        '''
        Initialization for EM algorithm.
        input:
            - X: data (batch_size, dim)
        '''
        self.data = X
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k), dtype=np.longdouble)
    
    def e_step(self):
        '''
        E-step of EM algorithm.
        '''
        z_reg = 1e-15
        for i in range(self.k):
            if np.isnan(self.sigma[i]).any():
                print(self.sigma[i], self.pi)
                input()
            self.z[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i]) + z_reg
        self.z /= self.z.sum(axis=1, keepdims=True)
    
    def m_step(self):
        '''
        M-step of EM algorithm.
        '''
        sum_z = self.z.sum(axis=0)
        self.pi = (sum_z / self.num_points + self.pi_reg) / self.pi.sum()

        self.mu = np.matmul(self.z.T, self.data)
        self.mu /= sum_z[:, None]
        for i in range(self.k):
            j = np.expand_dims(self.data, axis=1) - self.mu[i]
            
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.z[:, i] )
            self.sigma[i] /= sum_z[i]
            self.sigma[i] += self.covariance_reg_value * np.eye(self.dim)
        
    
    def run_em(self, max_iter):
        for i in range(max_iter):
            self.e_step()
            prev_sigma = np.copy(self.sigma)
            self.m_step()
            if np.linalg.norm(self.sigma - prev_sigma) < self.threshold_frob:
                ll = self.log_likelihood(self.data)
                return ll, i
        ll = self.log_likelihood(self.data)
        print('EM hasnt converged')
        return ll, i

        

    def log_likelihood(self, X):
        '''
        Compute the log-likelihood of X under current parameters
        input:
            - X: Data (batch_size, dim)
        output:
            - log-likelihood of X: Sum_n Sum_k log(pi_k * N( X_n | mu_k, sigma_k ))
        '''
        ll = []
        for d in X:
            tot = 0
            for i in range(self.k):
                tot += self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
            ll.append(np.log(tot))
        return np.sum(ll) / X.shape[0]
    
def precisions_to_sigmas_and_vice_versa(precisions):
    k = precisions.shape[0]
    sigmas = np.zeros_like(precisions)
    for k_ind in range(k):
        sigmas[k_ind] = np.linalg.inv(precisions[k_ind])
    return sigmas


if __name__ == '__main__':
    pass
    from utils import load_seg_data
    import time
    from sklearn.mixture import GaussianMixture
    data = load_seg_data()
    k = 5
    population_size = 100
    start_gmm = GaussianMixture(n_components=k, covariance_type='full', max_iter=2, n_init=population_size, init_params='k-means++')
    start_gmm.fit(data)
        
    weights = start_gmm.weights_
    means = start_gmm.means_
    precisions = start_gmm.precisions_

    max_iter = 300

    start_time = time.time()
    gmm_sklearn = GaussianMixture(n_components=weights.shape[0], covariance_type='full', weights_init=weights, means_init=means, precisions_init=precisions, max_iter=max_iter, verbose=1, verbose_interval=1)
    print(time.time() - start_time)
    gmm_sklearn.fit(data)
    print('GMM LL: ', gmm_sklearn.score(data))
    sigmas = precisions_to_sigmas_and_vice_versa(precisions)

    gmm_custom = []
    for n in range(population_size):

        start_time = time.time()
        gmm = GMM(weights.shape[0], dim=data.shape[1], init_mu=means, init_sigma=sigmas, init_pi=weights)

        gmm.init_em(data)

        converge_iter = gmm.run_em(max_iter)
        print(converge_iter)

        # for iter_n in range(max_iter):
        #     gmm.e_step()
        #     gmm.m_step()

        gmm_custom.append(gmm)
        # print(gmm.log_likelihood(data))
        print(time.time() - start_time)
