from default_config import *
from delays import *
from multiprocessing import Process, Lock
from sys import argv
import warnings
import multiprocessing
from navigation_env import *
import time

delay_keys = {
    "lognormal": TruncatedLogNormalDelay,
    "exp": TruncatedExponentialDelay,
    "const": ConstDelay,
}
save_config["directory"] = "mean"


def run_map(s, code):
    config["delay_kwargs"] = {"mean": 2.5, 'sigma': None, "skewness": s}
    save_config["experiment_key"] = config["delay_kwargs"]["skewness"]
    exec(code)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    lock = Lock()
    file_name = "heuristic.py"
    for s in [16, 12, 10, 8, 6, 4, 2]:
        with open(file_name, "r") as f:
            code = f.read()
        config["delay_kwargs"] = {"mean": s, 'sigma': None, "skewness": 2}
        save_config["experiment_key"] = config["delay_kwargs"]["mean"]
        exec(code)




