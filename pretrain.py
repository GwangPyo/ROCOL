import warnings
from stable_baselines import TRPO, PPO2, SAC, ACKTR, DDPG, TD3, ACER, DQN
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from navigation_env import NavigationEnvDefault
from default_config import config
from stable_baselines.gail import generate_expert_traj
from stable_baselines.gail import ExpertDataset

if __name__ == "__main__":
    env = SubprocVecEnv([lambda: NavigationEnvDefault(**config) for _ in range(32)])
    model = PPO2(env=env, policy="MlpLstmPolicy", n_steps=32, nminibatches=4, tensorboard_log='./', verbose=1)
    model.learn(1000000000)
    model.save("recurrent_nodelay")
