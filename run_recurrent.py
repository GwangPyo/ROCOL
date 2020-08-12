import warnings
warnings.simplefilter("ignore")
from stable_baselines import TRPO, PPO2, SAC, ACKTR, DDPG, TD3, ACER, DQN
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from navigation_env import *
import os
from default_config import config
from delays import *


def modify_action(exp_dataset):
    action_part = exp_dataset["actions"]
    a_map = [np.asarray([1, 0], dtype=np.float64), np.asarray([-1, 0], dtype=np.float64),
             np.asarray([0, 1], dtype=np.float64), np.asarray([0, -1], dtype=np.float64)]

    action_modified = [a_map[int(a)] for a in action_part]
    action_modified = np.asarray(action_modified, dtype=np.float32)
    exp_dataset["actions"] = action_modified


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    results = []

    for mean in [8, 10, 12, 14, 16]:
        # config["delay_kwargs"] = {"mean": s, "sigma":None, "skewness":s}
        n_cpu = 28
        print(config)
        config["delay_kwargs"]["mean"] = mean
        env = SubprocVecEnv([lambda: NavigationEnvPartialObs(**config) for _ in range(n_cpu)])
        model = PPO2.load("recurrent_nodelay.zip") # ("MlpLstmPolicy", env=env, n_steps=64, nminibatches=4, tensorboard_log="./", learning_rate=1e-5, ent_coef=1e-3 ) # PPO2.load("recurrent.zip", env=env)
        model.tensorboard_log = "./"
        model.set_env(env)
        model.verbose = 1
        model.learn(250000000)
        model.save("recurrent_delay_{}.zip".format(mean))
        scores = [ ]
        print(config)
        env = SubprocVecEnv([lambda: NavigationEnvPartialObs(**config) for _ in range(n_cpu)])
        model.set_env(env)
        # model.learn(1000000)
        obs = env.reset()
        state = None
        score_deque = []
        j = 0
        # for j in range(30000):
        while j < 10000:
            actions, state = model.predict(obs, state=state)
            obs, reward, done, info = env.step(actions)
            # reward = reward[0]
            if (done != False).any():
                j += 1

        for i in range(n_cpu):
            scores = scores + env.get_attr("last_score", i)
        with open("./dynamics/scores.csv", "a") as f:
            f.writelines("recurrent_{}".format(config["delay_kwargs"]["mean"]) + "," + str(np.mean(scores)) + "\n")

        env.close()
        del env

