from delays import *

config = {
    "max_speed": 5, "delay_function": TruncatedLogNormalDelay, "delay_kwargs": {"mean": 2, "sigma": 2},
    "max_obs_range": 3
}

save_config = {
    "directory": "deviation",
    "meta_policy_name": "ACKTR_MetaPolicy_{}",
    "Heuristic_Name": "Heuristic_{}",
    "experiment_key": config["delay_kwargs"]["sigma"]
}