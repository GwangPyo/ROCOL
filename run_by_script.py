from default_config import *
from delays import *
from multiprocessing import Process, Lock
from sys import argv
import warnings


delay_keys = {
    "lognormal": TruncatedLogNormalDelay,
    "exp": TruncatedExponentialDelay,
    "const": ConstDelay,
}
save_config["directory"] = "deviation"

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    lock = Lock()
    with open("run_meta.py", "r") as f:
        code = f.read()
    # for s in [2, 4, 6, 10, 12]:
    s = int(argv[1])
    config["delay_kwargs"]= {"mean": s, 'sigma': None, "skewness":s}
    save_config["experiment_key"] = config["delay_kwargs"]["skewness"]
    exec(code)


