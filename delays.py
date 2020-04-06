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

class TruncatedNormalDelay(NetworkDelay):
    def __init__(self, mean=10, sigma=1):
        super().__init__()
        assert mean > 1
        self.mu = mean - 1
        self.sigma = sigma ** 0.5

    def __call__(self):
        return np.random.normal(self.mu, self.sigma) + 1

class TruncatedGammaDelay(NetworkDelay):
    def __init__(self, mean=10, sigma=1):
        super().__init__()
        assert mean > 1
        self.scale = sigma/mean
        self.shape = mean/self.scale

    def __call__(self):
        return np.random.gamma(self.shape, self.scale)

class TruncatedStudentTDelay(NetworkDelay):
    def __init__(self, mean=10, sigma=1):
        super().__init__()
        assert mean > 1
        self.mean = mean
        if sigma == 1:
            self.nu = 10000
        else:
            self.nu = 2*sigma/(sigma-1)

    def __call__(self):
        return np.random.standard_t(self.nu) + self.mean

if __name__ == "__main__":
    # test
    delays = []
    normal = TruncatedNormalDelay(15, 3)
    for i in range(100000):
        delays.append(normal())
    print(np.mean(delays))
    print(np.var(delays))