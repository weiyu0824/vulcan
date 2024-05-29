import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error
from typing import List
import random

class BayesianOptimization:
    def __init__(self, feat_bounds: List[List[int]], acquisition='ucb'):
        """
        This BO is used to minimize cost 
        """
        
        for bound in feat_bounds:
            assert len(bound) != 0, 'feat bounds should not be empty'
        
        self.feat_bounds = feat_bounds
        self.num_feats = len(feat_bounds)
        self.acquisition = acquisition
        self.kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(random_state=0, kernel=self.kernel)
        self.y_min = 0

    def _get_acq_val(self, x, keppa=2):
        x = np.array(x)
        x = x.reshape(1, -1)
        if self.acquisition == 'ucb':
            mean, std = self.gp.predict(x, return_std=True)
            return mean + keppa * std
        elif self.acquisition == 'ei':
            mean, std = self.gp.predict(x, return_std=True)
            z = (self.y_min - mean) / std
            ei = (self.y_min - mean) * norm.cdf(z) + std * norm.pdf(z) 
            # print(mean, std, z, ei)

            # z = (y_pred - best_y) / y_std
            # ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
    
            return ei
        
    def get_pred(self, x):
        x = np.array(x)
        mean, std = self.gp.predict(x.reshape(1, -1), return_std=True)
        return mean, std 
    
    def _get_random_raw_samples(self, num_samples: int=300):
        samples = list()
        for _ in range(num_samples):
            feat_choices = [np.random.choice(bounds) for bounds in self.feat_bounds]
            samples.append(feat_choices)
        return samples

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit samples
        """
        self.gp = GaussianProcessRegressor(random_state=0, kernel=self.kernel).fit(X, y)
        self.y_min = min(y)
        # print(self.y_max)
        # exit()
        return True

    def get_next_sample(self):
        """Get a sample based on acqusition func"""
        samples = self._get_random_raw_samples()
        # best_acq_val = -10e100
        arr = []
        for sample in samples:
            acq_val = self._get_acq_val(sample)   
            arr.append((acq_val, sample))

        arr.sort(key=lambda x: x[0], reverse=True)
        # print('pred', arr[0][0])
        return arr[0][1]

    def get_sorted_samples(self):
        samples = self._get_random_raw_samples()
        # best_acq_val = -10e100
        arr = []
        for sample in samples:
            acq_val = self._get_acq_val(sample)   
            arr.append((acq_val, sample))

        arr.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in arr]
        # return random.choice(arr)[1]

if __name__ == "__main__":
    # Example usage:
    feat_bounds = [[0, 1, 2, 3, 4, 5], [0.2, 0.3, 0.4, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5]]  # Example bounds for categorical dimension

    bo = BayesianOptimization(feat_bounds)
    samples = bo.get_random_samples(3)

    y = [0, 1, 2]
    bo.fit(samples, y)
    
    best_sample = bo.get_next_sample()
    print(best_sample)