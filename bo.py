import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error
from typing import List

class BayesianOptimization:
    def __init__(self, cont_bounds: List[List[int]], cat_bounds, acquisition='ucb'):
        self.cont_bounds = np.array(cont_bounds)
        # print(self.cont_bounds)
        self.cat_bounds = cat_bounds
        self.num_continuous_feats = len(cont_bounds)
        self.num_categorical_feats = len(cat_bounds)
        self.acquisition = acquisition
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel)
    
    def _get_acq_val(self, x, keppa=2):
        if self.acquisition == 'ucb':
            x = x.reshape(1, -1)
            # print(x)
            # breakpoint()
            mean, std = self.gp.predict(x.reshape(1, -1), return_std=True)
            return mean + keppa * std
    
    def _get_random_raw_samples(self, num_samples: int):
        samples = []
        for _ in range(num_samples):
            cont_feats, cat_feats = [], []
            if len(self.cont_bounds) != 0:
                cont_feats = np.random.uniform(self.cont_bounds[:, 0], self.cont_bounds[:, 1])
            if len(self.cat_bounds) != 0:
                cat_feats = [np.random.choice(bounds) for bounds in self.cat_bounds]
            sample = np.hstack([cont_feats, cat_feats])
            samples.append(sample)
        return samples
    
    def get_random_samples(self, num_samples: int): 
        samples = self._get_random_raw_samples(num_samples)
        readable_samples = []
        for sample in samples:
            cont_feats = sample[:self.num_continuous_feats]
            cat_feats = sample[self.num_continuous_feats:] 
            readable_samples.append(
                {
                    'cont_feats': cont_feats,
                    'cat_feats': cat_feats
                }
            )       
        return readable_samples 

    def start(self):
        pass
        # Initialize with random points
        # self.X_continuous = np.random.uniform(self.bounds_continuous[:, 0], self.bounds_continuous[:, 1], size=(self.n_initial_points, self.n_continuous_dims))
        # self.X_categorical = [[np.random.choice(bounds) for bounds in self.bounds_categorical] for _ in range(self.n_initial_points)]



        # # breakpoint()
        # self.y = np.random.randn(self.n_initial_points)
        
        # # Initialize Gaussian Process
        # kernel = Matern(nu=2.5)
        # self.gp = GaussianProcessRegressor(kernel=kernel)
        # self.gp.fit(self.X_continuous, self.y)
        
        # improvement_counter = 0
        # best_utility = np.inf
        
        # while improvement_counter < self.consecutive_runs:
        #     next_continuous_point, next_categorical_point = self._get_next_query_point()
        #     utility_value = self._acquisition_function(next_continuous_point, next_categorical_point)
            
        #     # Perform actual evaluation of the objective function at the query point
        #     # Replace this line with the actual evaluation code
            
        #     # Example: 
        #     # new_y = objective_function(next_continuous_point, next_categorical_point)
        #     # self.X_continuous = np.vstack([self.X_continuous, next_continuous_point])
        #     # self.X_categorical.append(next_categorical_point)
        #     # self.y = np.append(self.y, new_y)
            
        #     if utility_value < best_utility - self.threshold:
        #         improvement_counter += 1
        #     else:
        #         improvement_counter = 0
        #         best_utility = utility_value
            
        #     self.iteration += 1
            
        # print("Bayesian Optimization stopped after {} consecutive runs without improvement.".format(self.consecutive_runs))

    def fit(self, X, y):
        """
        Fit samples
        """
        self.gp.fit(X, y)
        return True

    def get_next_sample(self):
        """Get a sample based on acqusition func"""
        samples = self._get_random_raw_samples(100)
        best_sample = None
        best_acq_val = 0
        for sample in samples:
            acq_val = self._get_acq_val(sample)        
            if acq_val > best_acq_val:
                best_sample = sample

        return best_sample

if __name__ == "__main__":
    # Example usage:
    bounds_continuous = np.array([[0, 10], [-5, 5]])  # Example bounds for continuous dimensions
    bounds_categorical = [[0, 1, 2]]  # Example bounds for categorical dimension
    bo = BayesianOptimization(bounds_continuous, bounds_categorical)
    # bo.start()

    samples = bo.get_random_samples(3)
    print(samples)
    # bo.fit()
    best_sample = bo.get_next_sample()
    print(best_sample)