#Author Francis Yu
#Acknowledgement to the MP2 code, the code logic is similar

import gym
import numpy as np
from pathlib import Path

# import envs
import logging
import time
import torch
from absl import app
from absl import flags
from policies import DQNPolicy, ActorCriticPolicy
from trainer_dqn import train_model_dqn
from evaluation import val, test_model_in_env

from torch import nn
from torchvision import transforms as T
from PIL import Image
from collections import deque
import random, datetime, os, copy

from env_preprocess import *
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 2, 'Number of episodes to evaluate.')
flags.DEFINE_integer('episode_len', 100, 'Length of each episode at test time.')
flags.DEFINE_string('env_name', 'mario', 'Name of environment.')
flags.DEFINE_boolean('vis', False, 'To visualize or not.')
flags.DEFINE_boolean('vis_save', False, 'To save visualization or not')
flags.DEFINE_integer('num_train_envs', 5, 'Number of the asynchronized environments')
flags.DEFINE_integer('seed', 1234, 'Seed for randomly initializing policies.')
flags.DEFINE_integer('input_frames', 4, 'The RGB frames to be stacked together.')
flags.DEFINE_integer('frame_shape', 84, 'The RGB frames shape (shape, shape).')
flags.DEFINE_float('gamma', 0.99, 'Discount factor gamma.')
flags.DEFINE_enum('algo', 'dqn', ['dqn', 'ac'], 'which algo to use, dqn or ac')
flags.DEFINE_string('logdir', 'debug', 'Directory to store loss plots, etc.')
flags.DEFINE_string('device', 'cuda', 'specifiy "cpu" if not using cuda')
flags.DEFINE_integer('use_ICM', 0, 'set 1 to use intrinsic reward module')
flags.DEFINE_integer('use_auxiliary', 0, 'set 1 to use auxiliary loss')
flags.DEFINE_boolean('auxiliary', True, 'True if use auxiliary tasks to improve training')

def make_env(env_name):
    if env_name == 'mario':
        env = gym_super_mario_bros.make("SuperMarioBros-v0")
        env = SkipFrame(env, skip=FLAGS.input_frames, return_auxiliary=FLAGS.auxiliary)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=FLAGS.frame_shape)
        env = FrameStack(env, num_stack=FLAGS.input_frames)
        return env
    elif env_name == 'lunar' or env_name == 'car':
        if env_name == 'lunar':
            env_name = 'LunarLander-v2'
        else:
            env_name = 'MountainCar-v0'
        env = gym.make(env_name).unwrapped
        env = GymFrameStack(env, FLAGS.input_frames, return_auxiliary=FLAGS.auxiliary)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=FLAGS.frame_shape)
        env = FrameStack(env, num_stack=FLAGS.input_frames)
        return env

    else:
        raise NotImplementedError(f'No environment named "{env_name}"')

def main(_):
    torch.manual_seed(FLAGS.seed)
    logdir = Path(FLAGS.logdir) / f'seed{FLAGS.seed}'
    logdir.mkdir(parents=True, exist_ok=True)

    # Setup training environments.
    train_envs = [make_env(FLAGS.env_name) for _ in range(FLAGS.num_train_envs)]
    [env.seed(i+FLAGS.seed) for i, env in enumerate(train_envs)]

    # Setting up validation environments.
    val_envs = [make_env(FLAGS.env_name) for _ in range(FLAGS.num_episodes)]
    [env.seed(i+1000) for i, env in enumerate(val_envs)]
    val_fn = lambda model, device: val(model, device, val_envs, FLAGS.episode_len)

    torch.set_num_threads(1)
    device = torch.device('cuda:0') if torch.cuda.is_available() and FLAGS.device == 'cuda' else torch.device('cpu')

    state_dim, action_dim = [FLAGS.input_frames, FLAGS.frame_shape, FLAGS.frame_shape], train_envs[0].action_space.n
    print(f'action size = {action_dim}')

    hidden_layers = [32, 32, 32, 32]
    ### Major training code using DQN
    if (FLAGS.algo == 'dqn') or (FLAGS.algo == 'dqn_noisy') or (FLAGS.algo == 'dqn_double') or (FLAGS.algo == 'dqn_all'):
        n_models = 1
        models, targets = [], []
        print('create learner network...')

        '''
        for both learner and target networks, specifiy the convolutional layers
        for the training. e.g. [32, 32, 32, 32] means four convolutional layers
        where each layer transform the channel dimension from: input_dim->32->32->32->32
        '''
        for i in range(n_models):
            models.append(DQNPolicy(state_dim, hidden_layers, action_dim, device))
            models[-1].to(device)
        
        print('create target network...')
        for i in range(n_models):
            targets.append(DQNPolicy(state_dim, hidden_layers, action_dim, device))
            targets[-1].to(device)

        train_model_dqn(models, targets, state_dim, action_dim, train_envs,
                        FLAGS.gamma, device, logdir, val_fn, FLAGS.use_ICM, env_name=FLAGS.env_name, use_auxiliary=FLAGS.use_auxiliary)
        model = models[0]

    elif FLAGS.algo == 'ac':
        raise NotImplementedError('AC not implemented yet!')
        # model = ActorCriticPolicy(state_dim, hidden_layers, action_dim)
        # train_model_ac(model, train_envs, FLAGS.gamma, device, logdir, val_fn, advantage=True)
    else:
        raise NotImplementedError('Only DQN/AC is implemented for not...')

    [env.close() for env in train_envs]
    [env.close() for env in val_envs]

if __name__ == '__main__':
    app.run(main)
