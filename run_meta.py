from navigation_env import NavigationEnvMeta, NavigationEnvMetaNoNet
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
import pickle
import numpy as np
import os
from stable_baselines.common.callbacks import EvalCallback
from delays import TruncatedLogNormalDelay
from default_config import config, save_config
import tensorflow as tf
import warnings

def meta_save_name(config_value):
    return save_config["directory"] + "/" + save_config["meta_policy_name"].format(config_value)


def log_save_name():
    return save_config["directory"] + "/"+ "scores.csv"


def eval_model(return_list,
        model_name, env_config, subpolicies = ("obs_range/ppo2_default_{}.zip".format(3),
                                                       "obs_range/ppo_local.zip"), env_type=NavigationEnvMeta,
               iteration=1000):
    subpolicies = list(subpolicies)
    environment = env_type(subpolicies=subpolicies, **env_config)
    model = PPO2.load(model_name)
    scores =[]
    for i in range(iteration):
        done = False
        obs = environment.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = environment.step(action)
            if done:
                if reward > 0:
                    scores.append(1)
                else:
                    scores.append(0)
    return_list.append(np.mean(scores))
    return


if __name__ == "__main__":
    # subpolicies = PPO2.load("test.zip")
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] ="-1"
    subpolicies = ["obs_range/ppo2_default_{}.zip".format(3), "obs_range/ppo_local.zip"]
    save_name = meta_save_name(config_value=save_config["experiment_key"])

    # config["delay_kwargs"]= {"mean": 2.5, 'sigma': None, "skewness":s}
    # save_config["experiment_key"] = config["delay_kwargs"]["skewness"]

    print(config)
    """
    env = SubprocVecEnv([lambda: NavigationEnvMeta(subpolicies, **config) for _ in range(8)])
    
    meta_policy = PPO2(policy="MlpPolicy", env=env, verbose=1, gamma=0.999, ent_coef=1e-4,
                        policy_kwargs={'act_fun':tf.nn.elu})

    meta_policy.learn(15000000, log_interval=5)
    
    try:
        meta_policy.save(save_name)
    except FileNotFoundError:
        os.mkdir(save_config["directory"])
        meta_policy.save(save_name)

    del meta_policy
    del env
    """
    meta_policy = PPO2.load(save_name)

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
        if i % 100 == 0:
            print(i ,"/", 10000)
            print("scores", np.mean(scores))

    # scores = env.timer_heuristic(episodes=10000)
    with open(log_save_name(), "a") as f:
        f.writelines("meta_{}".format(save_config["experiment_key"]) + "," + str(np.mean(scores)) + "\n")

    env.close()
    print(config)
    del env
