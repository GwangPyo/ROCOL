from navigation_env import NavigationEnvMeta, NavigationEnvMetaNoNet
from stable_baselines import PPO2, DQN, ACER, TRPO, ACKTR
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
import pickle
import numpy as np
import os
from stable_baselines.common.callbacks import EvalCallback
from delays import TruncatedLogNormalDelay

if __name__ == "__main__":
    # subpolicies = PPO2.load("test.zip")

    os.environ["CUDA_VISIBLE_DEVICES"] ="-1"
    subpolicies = ["obs_range/ppo2_default_{}.zip".format(3), "obs_range/ppo_local.zip"]
    config = {}

    for s in [2, 4, 6, 8, 10]:

        config["average_delay"] = 10
        config["max_speed"] = 5
        config["delay_function"] = TruncatedLogNormalDelay
        config["delay_kwargs"] = {"mean":10, "sigma":s}
        env = [lambda: NavigationEnvMeta(subpolicies=subpolicies, **config) for _ in range(8)]
        env = SubprocVecEnv(env)
        eval_env = NavigationEnvMeta(subpolicies=subpolicies, **config)
        eval_callback = EvalCallback(eval_env=eval_env, eval_freq=10000, deterministic=False, n_eval_episodes=20)

        meta_policy = ACKTR(policy="MlpPolicy", env=env, verbose=1, tensorboard_log='logs')
        meta_policy.learn(5000000, log_interval=3, callback=eval_callback)
        meta_policy.save("DelayDistribution/ACKTR_MetaPolicy_{}".format(config["delay_kwargs"]["sigma"]))
        env.close()
        del env
        del meta_policy

        meta_policy = ACKTR.load("DelayDistribution/ACKTR_MetaPolicy_{}".format(config["delay_kwargs"]["sigma"]))
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
            print(i ,"/", 1000)
        print("scores", np.mean(scores))
        with open("./obs_range/scores.csv", "a") as f:
            f.writelines("meta_{}".format(config["average_delay"]) + "," + str(np.mean(scores)) + "\n")

        with open("histogram_net_action", "wb") as f:
            pickle.dump(history, f)
        env.close()
        del env