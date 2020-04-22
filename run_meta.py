from navigation_env import NavigationEnvMeta, NavigationEnvMetaNoNet
from stable_baselines import PPO2, DQN, ACER, TRPO, ACKTR
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
import pickle
import numpy as np
import os
from stable_baselines.common.callbacks import EvalCallback
from delays import TruncatedLogNormalDelay
from default_config import config, save_config
from heuristic import NetworkHeuristic
import tensorflow as tf
import warnings

def meta_save_name(config_value):
    return save_config["directory"] + "/" + save_config["meta_policy_name"].format(config_value)


def log_save_name():
    return save_config["directory"] + "/"+ "scores.csv"


if __name__ == "__main__":
    # subpolicies = PPO2.load("test.zip")
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] ="-1"
    subpolicies = [ "obs_range/ppo_local.zip"]
    save_name = meta_save_name(config_value=save_config["experiment_key"])
    for s in [2]:
        print(config)
        env = SubprocVecEnv([lambda: NavigationEnvMeta(subpolicies=subpolicies, **config) for _ in range(8)])
        eval_env = NavigationEnvMeta(subpolicies=subpolicies, **config)
        eval_callback = EvalCallback(eval_env=eval_env, eval_freq=100000, deterministic=False, n_eval_episodes=200)
        meta_policy = ACKTR(policy="MlpPolicy", env=env, verbose=1, gamma=0.999, ent_coef=1e-4,
                            policy_kwargs={'act_fun':tf.nn.elu}, lr_schedule='middle_drop')
        meta_policy.learn(5000000, log_interval=5, callback=eval_callback)

        try:
            meta_policy.save(save_name)
        except FileNotFoundError:
            os.mkdir(save_config["directory"])
            meta_policy.save(save_name)

        del meta_policy
        del env

        meta_policy = ACKTR.load(save_name)
        env = NavigationEnvMeta(subpolicies=subpolicies, **config)
        history = []
        scores = []

        for i in range(10000):
            obs = env.reset()

            done = False
            while not done:
                # env.render()
                dict_obs =  NavigationEnvMeta.dict_observation(env)
                action, _ = meta_policy.predict(obs)
                x = dict_obs["network_state"]
                y = action
                obs, reward, done, info = env.step(action)
                history.append([x, y])
                if done:
                    if reward > 0:
                        scores.append(1)
                    else:
                        scores.append(0)
            print(i ,"/", 100)
            print("scores", np.mean(scores))

        with open(log_save_name(), "a") as f:
            f.writelines("meta_{}".format(save_config["experiment_key"]) + "," + str(np.mean(scores)) + "\n")

        with open("histogram_net_action", "wb") as f:
            pickle.dump(history, f)
        env.close()
        print(config)
        del env