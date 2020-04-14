import warnings
warnings.simplefilter("ignore")

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import LnMlpPolicy
from stable_baselines import TRPO, PPO2, SAC, ACKTR, DDPG, TD3, ACER, DQN
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.common.callbacks import EvalCallback

from navigation_env import NavigationEnvDefault, NavigationEnvWall, NavigationEnvMeta
from stable_baselines.gail import ExpertDataset, generate_expert_traj
import os
import tensorflow as tf
import numpy as np
from multiprocessing import *
from delays import *


def modify_action(exp_dataset):
    action_part = exp_dataset["actions"]
    a_map = [np.asarray([1, 0], dtype=np.float64), np.asarray([-1, 0], dtype=np.float64),
             np.asarray([0, 1], dtype=np.float64), np.asarray([0, -1], dtype=np.float64)]

    action_modified = [a_map[int(a)] for a in action_part]
    action_modified = np.asarray(action_modified, dtype=np.float32)
    exp_dataset["actions"] = action_modified


config = {
    "max_speed":5
}




if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    results = []

    config["max_speed"] = 5
    config["max_obs_range"] = 3
    config["delay_function"] = TruncatedLogNormalDelay
    for s in [2, 4, 6, 8, 10, 16]:
        config["delay_kwargs"] = {"mean": 2, "skewness":s, "sigma":None}
        n_cpu = 32

        # model = PPO2.load("./obs_range/ppo2_default_{}.zip".format(config["max_obs_range"])) # icies=["logs/best_model.zip"], **config) for _ in range(n_cpu)])
        subpolicies = ["obs_range/ppo2_default_{}.zip".format(3)]
        env = SubprocVecEnv([lambda: NavigationEnvMeta( subpolicies= subpolicies, **config) for _ in range(n_cpu)])
        model = PPO2(policy="MlpPolicy", env=env)
        scores = [ ]
        obs = env.reset()

        for j in range(30000):
            actions, _ = model.predict(obs)
            obs, reward, done, info = env.step(actions)
            if j % 100 == 0:
                print(j, "/", 30000)
        for i in range(n_cpu):
            scores = scores + env.get_attr("last_score", i)
        with open("./skewness/scores.csv", "a") as f:
            f.writelines("local" + "," + str(np.mean(scores)) + "\n")

        env.close()
        del env

