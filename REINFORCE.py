import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import torch.optim as optim
from torch.distributions.normal import Normal


class IntrinsicRewardNetwork(object):
    def __init__(self, obs_shape):
        self.obs_shape= obs_shape

        self.obs_target = nn.Sequential(
            nn.Linear(self.obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.obs_regression = nn.Sequential(
            nn.Linear(self.obs_shape, 24),
            nn.ReLU(),
            nn.Linear(24, 4)
        )
        self.optimizer = optim.SGD(params=self.obs_target.parameters(), lr=1e-3, momentum=0.9)
        self.criterion = nn.MSELoss()

    def intrinsic_reward(self, obs):
        obs = torch.from_numpy(obs)
        y_true = self.obs_target(obs)
        y_pred = self.obs_regression(obs)
        loss = self.criterion(y_true, y_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class Reinforce(nn.Module):
    def __init__(self, env, gamma=0.99, lr=1e-4):
        super(Reinforce, self).__init__()
        self.gamma = gamma
        self.env = env
        self.obs_shape = np.prod(env.observation_space.shape, axis=None)
        self.action_space = self.env.action_space
        self.action_shape = env.action_space.shape
        self.action_size = np.prod(env.action_space.shape)
        # observation embedding
        self.obs_embedding = nn.Sequential(
            nn.Linear(self.obs_shape, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )
        self.mean_layer =  nn.Linear(64, self.action_size)
        self.log_dev_layer = nn.Linear(64, self.action_size)
        self._runner = None
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.critic_criterion = torch.nn.MSELoss()

    def _build_runner(self):
        self._runner = Runner(self)

    def forward(self, x):
        x = self.obs_embedding(x)

        mean = self.mean_layer(x)
        log_sigma = self.log_dev_layer(x)
        sigma = F.softplus(log_sigma + 1e-5)

        return mean, sigma

    def predict(self, state):
        state = torch.flatten(torch.from_numpy(state).float())
        mean, sigma = self(state)

        action_dist = Normal(mean, sigma)
        action = action_dist.sample()
        action = action.view(self.action_shape)

        return np.clip(action.numpy(), self.action_space.low, self.action_space.high)

    def step(self, state):
        state = torch.flatten(torch.from_numpy(state).float())
        mean, sigma = self(state)

        dist = Normal(mean, sigma)
        action = dist.sample()
        action = action.view(self.action_shape)
        print("log dist", dist.log_prob(action))
        normal_dist = torch.normal(mean, sigma)
        prob = torch.normal(action, mean, sigma)
        print("log prob", torch.log(prob))
        return action.numpy(), dist.log_prob(action)

    def train(self, num_episodes):
        if self._runner is None:
            self._build_runner()
        episode_loss = 0
        eps = 1e-3
        for i in range(num_episodes):
            obs, log_action_prob, returns, advantage, episode_rewards = self._runner.rollout()
            returns = torch.FloatTensor(returns)
            # baseline as a standardize of returns
            std_return = returns.std()
            if std_return == std_return:
                returns = (returns - returns.mean()) / (returns.std() + eps)
            # Policy loss = Expectation( gradient (log pi_theta (s, a) * r))

            mean_loss = []
            for log_prob, R in zip(log_action_prob, returns):
                loss = -log_prob * R
                loss = loss.mean()
                self.optimizer.zero_grad()
                self.optimizer.step()
                mean_loss.append(loss)
            mean_loss = torch.FloatTensor(mean_loss)
            mean_loss = mean_loss.mean()
            print("==========================================")
            print("|", end='')
            print("mean loss:\t{:4f}".format(mean_loss.item()), end='|\n')
            print("|", end='')
            print("episode reward:\t{:.4f}".format(episode_rewards), end='|\n')
            print("==========================================")

    def set_env(self, env):
        self.env = env

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, *args, **kwargs):
        model = cls(*args, **kwargs)
        model.load_state_dict(path)
        return model


class Runner(object):
    def __init__(self, policy):
        self.policy = policy
        self.gamma = self.policy.gamma
        self.env = self.policy.env
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high
        self.int_r = IntrinsicRewardNetwork(self.policy.obs_shape)

    def rollout(self):
        done = False
        states = []
        log_probs = []
        rewards = []
        predicted_value = []
        obs = self.env.reset()
        while not done:
            # save obs
            states.append(obs)
            # predict action
            action, log_prob = self.policy.step(obs)
            clip_action = np.clip(action, self.low, self.high)
            # save action
            log_probs.append(log_prob)
            # step env
            obs, reward, done, info = env.step(clip_action)
            # save rewards
            rewards.append(reward)


        # compute return
        returns = self.rewards_to_return(rewards)
        # self.intrinsic_rewards(states, returns)
        return states, log_probs, returns, predicted_value, np.sum(rewards)

    def intrinsic_rewards(self, obs, returns):
        for i in range(len(obs)):
            o = obs[i]
            o = np.asarray(o, dtype=np.float32)
            returns[i] += self.int_r.intrinsic_reward(o)

    def rewards_to_return(self, rewards):
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns


if __name__ == "__main__":
    from navigation_env import *
    from default_config import config
    env = gym.make("LunarLanderContinuous-v2")# NavigationEnvDefault(**config)
    policy = Reinforce(env, lr=1e-4)
    policy.train(num_episodes=1000000)