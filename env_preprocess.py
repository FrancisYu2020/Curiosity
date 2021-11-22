import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym import spaces 

class GymFrameStack(gym.Wrapper):
    def __init__(self, env, k, run_with_video=True):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.run_with_video = run_with_video

    def reset(self):
        ob = self.env.reset()
        ob = self.env.render(mode='rgb_array')
        return ob
        # for _ in range(self.k):
        #     self.frames.append(ob)
        # return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # self.frames.append(self.env.render(mode='rbg_array'))
        return self.env.render(mode='rgb_array'), reward, done, info

    # def _get_ob(self):
    #     states = np.array(self.frames)
    #     print(states.shape)
    #     return np.concatenate(states,axis=1)

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip, run_with_video=True):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.run_with_video = run_with_video

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
            if self.run_with_video:
                self.env.render()
        return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation