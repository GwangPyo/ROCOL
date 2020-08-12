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


def run_map(s, code):
    config["delay_kwargs"] = {"mean": 2.5, 'sigma': None, "skewness": s}
    save_config["experiment_key"] = config["delay_kwargs"]["skewness"]
    exec(code)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    lock = Lock()
    file_name = "heuristic.py"

    for s in [2, 4, 6, 8, 10, 12, 14, 16]:
        for dyn in [5]:
            with open(file_name, "r") as f:
                code = f.read()
            save_config["directory"] = "skewness"
            config["delay_kwargs"] = {"mean": 2, 'sigma': None, "skewness": s}
            config["max_speed"] = dyn
            save_config["experiment_key"] = s
            exec(code)



