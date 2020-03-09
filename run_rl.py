from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO, PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from navigation_env import NavigationEnvDefault, NavigationEnvWall


if __name__ == "__main__":
    env = SubprocVecEnv([lambda: NavigationEnvWall() for _ in range(12)])
    model = PPO2(env=env, policy=MlpPolicy, verbose=1)
    model.learn(50000000)
    model.save("test")
    env = NavigationEnvDefault()
    done = False
    obs = env.reset()
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

