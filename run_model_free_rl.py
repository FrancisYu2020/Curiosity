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
from trainer_ac import train_model_ac
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
flags.DEFINE_integer('num_episodes', 10, 'Number of episodes to evaluate.')
flags.DEFINE_integer('episode_len', 200, 'Length of each episode at test time.')
flags.DEFINE_string('env_name', 'mario', 'Name of environment.')
flags.DEFINE_boolean('vis', False, 'To visualize or not.')
flags.DEFINE_boolean('vis_save', False, 'To save visualization or not')
flags.DEFINE_integer('num_train_envs', 5, 'Number of the asynchronized environments')
flags.DEFINE_integer('seed', 0, 'Seed for randomly initializing policies.')
flags.DEFINE_integer('input_frames', 4, 'The RGB frames to be stacked together.')
flags.DEFINE_integer('frame_shape', 84, 'The RGB frames shape (shape, shape).')
flags.DEFINE_float('gamma', 0.99, 'Discount factor gamma.')
flags.DEFINE_enum('algo', 'dqn', ['dqn', 'dqn_double', 'dqn_noisy', 'dqn_all', 'ac'], 'which algo to use, dqn or ac')
flags.DEFINE_string('logdir', 'debug', 'Directory to store loss plots, etc.')
flags.DEFINE_boolean('use_ICM', True, 'set True to use intrinsic reward module')
# flags.mark_flag_as_required('logdir')
# flags.mark_flag_as_required('algo')

def make_env(env_name):
    if env_name == 'mario':
        env = gym_super_mario_bros.make("SuperMarioBros-v0")
        # Apply Wrappers to environment
        env = SkipFrame(env, skip=1)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=FLAGS.frame_shape)
        env = FrameStack(env, num_stack=FLAGS.input_frames)
        return env
    elif env_name == 'lunar' or env_name == 'car':
        return gym.make(env_name)
    else:
        raise NotImplementedError(f'No environment named "{env_name}"')

# def get_dims(env_name):
#     if env_name == 'mario':
#         return 4, 2
#     elif env_name == 'lunar':
#         return None
#     elif env_name == 'car':
#         return None
#     else:
#         raise NotImplementedError(f'No environment named "{env_name}"')

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
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # state_dim, action_dim = get_dims(FLAGS.env_name)
    state_dim, action_dim = [FLAGS.input_frames, FLAGS.frame_shape, FLAGS.frame_shape], train_envs[0].action_space.n

    if (FLAGS.algo == 'dqn') or (FLAGS.algo == 'dqn_noisy') or (FLAGS.algo == 'dqn_double') or (FLAGS.algo == 'dqn_all'):
        noisy = (FLAGS.algo[-5:] == 'noisy')
        double = (FLAGS.algo[-6:] == 'double')
        if FLAGS.algo[-3:] == 'all':
            noisy, double = True, True
        n_models = 1
        models, targets = [], []
        print('create learner network...')
        for i in range(n_models):
            models.append(DQNPolicy(state_dim, [32], action_dim, device))
            models[-1].to(device)
        
        print('create target network...')
        for i in range(n_models):
            targets.append(DQNPolicy(state_dim, [32], action_dim, device))
            targets[-1].to(device)

        train_model_dqn(models, targets, state_dim, action_dim, train_envs,
                        FLAGS.gamma, device, logdir, val_fn, double, noisy, FLAGS.use_ICM)
        model = models[0]

    elif FLAGS.algo == 'ac':
        model = ActorCriticPolicy(state_dim, [32, 32], action_dim)
        train_model_ac(model, train_envs, FLAGS.gamma, device, logdir, val_fn, advantage=False)
    # elif FLAGS.algo == 'a2c':
    #     model = ActorCriticPolicy(state_dim, [16, 32, 64], action_dim)
    #     train_model_ac(model, train_envs, FLAGS.gamma, device, logdir, val_fn, advantage=True)
    else:
        raise NotImplementedError('Only DQN/AC is implemented for not...')

    [env.close() for env in train_envs]
    [env.close() for env in val_envs]

    if FLAGS.vis or FLAGS.vis_save:
        env_vis = make_env(FLAGS.env_name)
        state, g, gif, info = test_model_in_env(
            model, env_vis, FLAGS.episode_len, device, vis=FLAGS.vis,
            vis_save=FLAGS.vis_save)
        if FLAGS.vis_save:
            gif[0].save(fp=f'{logdir}/vis-{env_vis.unwrapped.spec.id}.gif',
                        format='GIF', append_images=gif,
                        save_all=True, duration=50, loop=0)
        env_vis.close()

if __name__ == '__main__':
    app.run(main)
