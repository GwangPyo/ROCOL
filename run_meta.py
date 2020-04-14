from navigation_env import NavigationEnvMeta, NavigationEnvMetaNoNet
from stable_baselines import PPO2, DQN, ACER, TRPO, ACKTR
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
import pickle
import numpy as np
import os
from stable_baselines.common.callbacks import EvalCallback
from delays import TruncatedLogNormalDelay
from heuristic import NetworkHeuristic


save_config = {
    "directory": "mean",
    "meta_policy_name":"ACKTR_MetaPolicy_{}",
    "Heuristic_Name":"Heuristic_{}"
}


def meta_save_name(config_value):
    return save_config["directory"] + "/" + save_config["meta_policy_name"].format(config_value)


def log_save_name():
    return save_config["directory"] + "/"+ "scores.csv"


if __name__ == "__main__":
    # subpolicies = PPO2.load("test.zip")

    os.environ["CUDA_VISIBLE_DEVICES"] ="-1"
    subpolicies = ["obs_range/ppo2_default_{}.zip".format(3), "obs_range/ppo_local.zip"]
    config = {}

    for s in [2, 4, 6, 8, 10]:

        config["max_speed"] = 5
        config["delay_function"] = TruncatedLogNormalDelay
        config["delay_kwargs"] = {"mean": s, "sigma": 4}

        save_name = meta_save_name(config_value=config["delay_kwargs"]["mean"])

        env = [lambda: NavigationEnvMeta(subpolicies=subpolicies, **config) for _ in range(8)]
        env = SubprocVecEnv(env)
       
        eval_env = NavigationEnvMeta(subpolicies=subpolicies, **config)
        eval_callback = EvalCallback(eval_env=eval_env, eval_freq=10000, deterministic=False, n_eval_episodes=20)

        meta_policy = ACKTR(policy="MlpPolicy", env=env, verbose=1, tensorboard_log='logs')
        meta_policy.learn(5000000, log_interval=3, callback=eval_callback)

        try:
            meta_policy.save(save_name)
        except FileNotFoundError:
            os.mkdir(save_config["directory"])
            meta_policy.save(save_name)

        del meta_policy
        env.close()
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
            f.writelines("meta_{}".format(config["delay_kwargs"]["mean"]) + "," + str(np.mean(scores)) + "\n")

        with open("histogram_net_action", "wb") as f:
            pickle.dump(history, f)
        env.close()
        del env