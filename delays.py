import numpy as np
from abc import ABCMeta
from abc import abstractmethod


class NetworkDelay(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self):
        pass


class TruncatedExponentialDelay(NetworkDelay):
    def __init__(self, mean=10):
        super().__init__()
        assert mean > 1
        self.scale = mean - 1

    def __call__(self):
        return np.random.exponential(self.scale) + 1


class TruncatedLogNormalDelay(NetworkDelay):
    def __init__(self, mean=10, sigma=1):
        super().__init__()
        assert np.log(mean) > 1
        self.mu = np.log(mean) - 1
        self.sigma = sigma

    def __call__(self):
        return np.random.lognormal(self.mu, self.sigma) + 1
