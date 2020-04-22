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
    def __init__(self, mean=10, sigma=4, skewness=None):
        super().__init__()
        if skewness is not None and sigma is not None:
            raise AttributeError("sigma and skewness cannot given at the same time")
        if skewness is None and sigma is None:
            raise AttributeError("one of the value sigma and skewness must be given")
        if sigma is not None:
            self.sigma = np.sqrt(np.log(np.square(sigma)/np.square(mean) + 1))
        else:
            self.sigma = TruncatedLogNormalDelay.skew_to_sigma(skewness)
        self.mu = np.log(mean) - np.square(self.sigma)/2

    def __call__(self):
        return np.random.lognormal(self.mu, self.sigma) + 1

    @staticmethod
    def skew_to_sigma(skewness):
        def auxiliary_function(x):
            return np.sqrt(x ** 2 + np.sqrt(x ** 4 + 4 * x ** 2 ) + 2)
        const = np.power(2, 1/3)
        exp_sigma_square = auxiliary_function(skewness)/const + const/auxiliary_function(skewness) - 1
        return np.sqrt(np.log(exp_sigma_square))


class ConstDelay(NetworkDelay):
    def __init__(self, mean):
        super().__init__()
        self.val = mean

    def __call__(self):
        return self.val


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    lists = []
    delay = TruncatedLogNormalDelay(mean=2, sigma=2)

    while len(lists) < 1000000:
        lists.append(delay())
    print(np.mean(lists))
    print(np.std(lists))
    plt.hist(lists, bins=np.arange(1, 10))
    plt.show()