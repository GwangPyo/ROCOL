from navigation_env import NavigationEnvMeta, NavigationEnvHeuristic
from stable_baselines import PPO2, DQN, ACER, TRPO, ACKTR
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
import pickle
import numpy as np
import os
from stable_baselines.common.callbacks import EvalCallback
from delays import TruncatedLogNormalDelay
from default_config import config, save_config
import tensorflow as tf
import warnings
import scipy.optimize

def meta_save_name(config_value):
    return save_config["directory"] + "/" + "timer_heuristic{}".format(config_value)


def log_save_name():
    return save_config["directory"] + "/"+ "scores.csv"


def reservation_heurstic(obs, thresh_hold = 20):
    if obs > thresh_hold:
        return 1
    else:
        return 0


def heuristic_eval(x):

    def heuristic(x):
        def inner(obs):
            return reservation_heurstic(obs, thresh_hold=x)
        return inner

    scores = []
    subpolicies = ["obs_range/ppo2_default_{}.zip".format(3),  "obs_range/ppo_local.zip"]
    env = NavigationEnvMeta(subpolicies=subpolicies, **config)

    func = heuristic(x)
    for _ in range(100):
        env.reset()
        done = False
        while not done:
            # env.render()
            dict_obs = NavigationEnvMeta.dict_observation(env)
            obs = dict_obs["network_state"] * 10
            action = func(obs)

            obs, reward, done, info = env.step(action)
            if done:
                if reward > 0:
                    scores.append(1)
                else:
                    scores.append(0)

    return -np.mean(scores)


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    subpolicies = ["obs_range/ppo2_default_{}.zip".format(3),  "obs_range/ppo_local.zip"]
    save_name = meta_save_name(config_value=save_config["experiment_key"])

    # config["delay_kwargs"]= {"mean": 2.5, 'sigma': None, "skewness":s}
    # save_config["experiment_key"] = config["delay_kwargs"]["skewness"]

    print(config)

    env = NavigationEnvMeta(subpolicies=subpolicies, **config)
    """
    history = []
    scores = []

    for i in range(10000):
        env.reset()
        done = False
        while not done:
            # env.render()
            dict_obs = NavigationEnvMeta.dict_observation(env)
            obs = dict_obs["network_state"] * 10
            action = reservation_heurstic(obs)

            obs, reward, done, info = env.step(action)
            if done:
                if reward > 0:
                    scores.append(1)
                else:
                    scores.append(0)
        if i % 100 == 0:
            print(i, "/", 10000)
            print("scores", np.mean(scores))

    # scores = env.timer_heuristic(episodes=10000)
    with open(log_save_name(), "a") as f:
        f.writelines("heuristic_{}".format(save_config["experiment_key"]) + "," + str(np.mean(scores)) + "\n")

    env.close()
    print(config)
    del env
    """

    print(scipy.optimize.minimize(heuristic_eval, x0=np.asarray([13])))