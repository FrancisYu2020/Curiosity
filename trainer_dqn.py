#TODO: finishing adding auxiliary loss in this file


import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import warnings
from copy import deepcopy
from model import *

num_steps_per_rollout = 50
num_updates = 1000
reset_every = 200
val_every = 2000

replay_buffer_size = 1000
q_target_update_every = 50
q_batch_size = 1024
q_num_steps = 5

def log(writer, iteration, name, value, print_every=10, log_every=10):
    # A simple function to let you log progress to the console and tensorboard.
    if np.mod(iteration, print_every) == 0:
        if name is not None:
            print('{:8d}{:>30s}: {:0.3f}'.format(iteration, name, value))
        else:
            print('')
    if name is not None and np.mod(iteration, log_every) == 0:
        writer.add_scalar(name, value, iteration)

# Implement a replay buffer class that a) stores rollouts as they
# come along, overwriting older rollouts as needed, and b) allows random
# sampling of transition quadruples for training of the Q-networks.
class ReplayBuffer(object):
    def __init__(self, size, state_dim, action_dim, device, env_name='mario'):
        self.size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.s = torch.zeros([size] + state_dim)
        self.a = np.zeros((size, action_dim))
        self.s_prime = np.zeros([size] + state_dim)
        self.r = torch.zeros(size).to(device)
        self.done = torch.zeros(size).to(device)
        if env_name == 'mario':
            self.info = np.zeros((size, 1))
        elif env_name == 'car':
            self.info = np.zeros((size, 2))
        elif env_name == 'lunar':
            self.info = np.zeros((size, 8))
        else:
            raise NotImplementedError(f'{env_name} info is not retrivable!')
        self.curr_size = 0

    def insert(self, rollouts):
        offset = max(0, self.curr_size + len(rollouts) - self.size)
        if offset:
            self.s[:(self.size - offset), :] = self.s[offset:, :].clone()
            self.a[:(self.size - offset), :] = self.a[offset:, :]
            self.s_prime[:(self.size - offset), :] = self.s_prime[offset:, :]
            self.r[:(self.size - offset)] = self.r[offset:].clone()
            self.done[:(self.size - offset)] = self.done[offset:].clone()
            self.info[:(self.size - offset)] = self.info[offset:]
            for i in range(len(rollouts)):
                k = i-len(rollouts)
                self.s[k, :], self.a[k, :], self.s_prime[k,:], self.r[k], self.done[k], self.info[k, :] = rollouts[i]
            self.curr_size = self.size
        else:
            for i in range(len(rollouts)):
                k = self.curr_size
                self.s[k, :], self.a[k, :], self.s_prime[k,:], self.r[k], self.done[k], self.info[k, :] = rollouts[i]
                self.curr_size += 1

    def sample_batch(self, batch_size):
        samples = None
        # TODO
        if batch_size > self.curr_size:
            warnings.warn('Not enough rollouts to sample')
            k = self.curr_size
            return self.s[:k,:], self.a[:k,:], self.s_prime[:k,:], self.r[:k], self.done[:k], self.info[:k]
        k = np.random.choice(self.curr_size, batch_size, replace=False)
        return self.s[k], self.a[k], self.s_prime[k], self.r[k], self.done[k], self.info[k]

# Starting off from states in envs, rolls out num_steps_per_rollout for each
# environment using the policy in `model`. Returns rollouts in the form of
# states, actions, rewards and new states. Also returns the state the
# environments end up in after num_steps_per_rollout time steps.
def collect_rollouts(models, envs, states, num_steps_per_rollout, epsilon, device, return_auxiliary=False):
    rollouts = []
    # TODO
    for i in range(num_steps_per_rollout):
        actions = models[-1].act(states, epsilon).cpu().numpy().squeeze()
        for j in range(len(actions)):
            rollouts.append([states[j], actions[j]] + list(envs[j].step(actions[j])))
            rollouts[-1][-2] = bool(rollouts[-1][-2])
        states = []
        collected_rollouts = rollouts[-len(envs):]
        for k in range(len(envs)):
            if not collected_rollouts[k][-2]:
                states.append(collected_rollouts[k][2])
            else:
                states.append(envs[k].reset())
        states = torch.from_numpy(np.array(states)).float().to(device)
    # print(rollouts[-1][-1])
    # exit(0)
    return rollouts, states

# Function to train the Q function. Samples q_num_steps batches of size
# q_batch_size from the replay buffer, runs them through the target network to
# obtain target values for the model to regress to. Takes optimization steps to
# do so. Returns the bellman_error for plotting.
def update_model(replay_buffer, models, targets, optim, gamma, action_dim,
                 q_batch_size, q_num_steps, device, double, ICM_module, optim_ICM, reinforce=0.1):
    total_bellman_error = 0.
    for step in range(q_num_steps):
        optim.zero_grad()
        sample = replay_buffer.sample_batch(q_batch_size)
        s, a, s_prime, r, done, info = sample
        s, r, s_prime = s.to(device), r.to(device), torch.from_numpy(s_prime).float().to(device)

        if not double:
            #target for vanilla DQN
            y = r + gamma*targets[-1](s_prime).max(dim=1)[0]
        else:
            #target for double q-learning
            y = r + gamma*torch.take_along_dim(targets[-1](s_prime), models[-1](s_prime).argmax(dim=1, keepdim=True), dim=1).view(-1)
        pred_qvals = torch.take_along_dim(models[-1](s), torch.from_numpy(a).long().to(device), dim=1).view(-1)
        step_bellman_error = torch.nn.functional.mse_loss(pred_qvals, y, reduction='mean')
        if ICM_module is not None:
            step_bellman_error = reinforce * step_bellman_error + ICM_module.intrinsic_loss(a, s, s_prime)
        step_bellman_error.backward()
        total_bellman_error += step_bellman_error.item()
        optim.step()
    return total_bellman_error / q_num_steps

def train_model_dqn(models, targets, state_dim, action_dim, envs, gamma, device, logdir, val_fn, double, use_ICM, reinfoce=0.1):
    train_writer = SummaryWriter(logdir / 'train')
    val_writer = SummaryWriter(logdir / 'val')

    # You may want to setup an optimizer, loss functions for training.
    # optim = torch.optim.RMSprop(models[-1].parameters())
    optim = torch.optim.Adam(models[-1].parameters(), lr=1.5e-3)
    ICM_module = ICM(state_dim, action_dim=action_dim).to(device) if use_ICM else None
    optim_ICM = torch.optim.Adam(ICM_module.parameters(), lr=1.5e-3) if use_ICM else None

    # Set up the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, state_dim, 1, device)

    # Resetting all environments to initialize the state.
    num_steps, total_samples = 0, 0
    states = torch.from_numpy(np.array([e.reset() for e in envs])).float().to(device)

    for updates_i in range(num_updates):
        # Come up with a schedule for epsilon
        epsilon = max(0.9*(1 - 2*updates_i/num_updates) + 0.1, 0.1)

        # Put model in training mode.
        [m.train() for m in models]

        if np.mod(updates_i, q_target_update_every) == 0:
            # If you are using a target network, every few updates you may want
            # to copy over the model to the target network.
            # TODO
            for i in range(len(models)):
                targets[i] = deepcopy(models[i])
                targets[i].eval()

        # Collect rollouts using the policy.
        rollouts, states = collect_rollouts(models, envs, states, num_steps_per_rollout, epsilon, device)
        num_steps += num_steps_per_rollout
        total_samples += num_steps_per_rollout*len(envs)

        # Push rollouts into the replay buffer.
        replay_buffer.insert(rollouts)


        # Use replay buffer to update the policy and take gradient steps.
        bellman_error = update_model(replay_buffer, models, targets, optim,
                                     gamma, action_dim, q_batch_size,
                                     q_num_steps, device, double, ICM_module, optim_ICM, reinfoce)
        print(updates_i, total_samples)
        log(train_writer, updates_i, 'train-samples', total_samples, 100, 1)
        log(train_writer, updates_i, 'train-bellman-error', bellman_error, 100, 1)
        log(train_writer, updates_i, 'train-epsilon', epsilon, 100, 1)
        log(train_writer, updates_i, None, None, 100, 1)


        # We are solving a continuing MDP which never returns a done signal. We
        # are going to manully reset the environment every few time steps. To
        # track progress on the training envirnments you can maintain the
        # returns on the training environments, and log or print it out when
        # you reset the environments.
        # if num_steps >= reset_every:
        #     states = torch.from_numpy(np.array([e.reset() for e in envs])).float().to(device)
        #     num_steps = 0
        #     if noisy:
        #         for m in models:
        #             m.reset_noise()

        # Every once in a while run the policy on the environment in the
        # validation set. We will use this to plot the learning curve as a
        # function of the number of samples.
        cross_boundary = total_samples // val_every > \
            (total_samples - len(envs)*num_steps_per_rollout) // val_every
        if cross_boundary:
            models[0].eval()
            mean_reward = val_fn(models[0], device)
            log(val_writer, total_samples, 'val-mean_reward', mean_reward, 1, 1)
            log(val_writer, total_samples, None, None, 1, 1)
            models[0].train()
