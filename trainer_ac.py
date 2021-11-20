from tensorboardX import SummaryWriter
import numpy as np
import torch

num_steps_per_rollout = 5
num_updates = 10000
reset_every = 200
val_every = 1000


def log(writer, iteration, name, value, print_every=10, log_every=10):
    # A simple function to let you log progress to the console and tensorboard.
    if np.mod(iteration, print_every) == 0:
        if name is not None:
            print('{:8d}{:>30s}: {:0.3f}'.format(iteration, name, value))
        else:
            print('')
    if name is not None and np.mod(iteration, log_every) == 0:
        writer.add_scalar(name, value, iteration)

# Starting off from states in envs, rolls out num_steps_per_rollout for each
# environment using the policy in `model`. Returns rollouts in the form of
# states, actions, rewards and new states. Also returns the state the
# environments end up in after num_steps_per_rollout time steps.
def collect_rollouts(model, env, states, num_steps, device):
    rollouts = [None for _ in range(len(envs)*num_steps + 1)]
    # TODO
    for i in range(num_steps):
        actions = model.act(states, sample=True).cpu().numpy()
        for j in range(len(actions)):
            rollouts[i+num_steps*j] = [states[j], actions[j]] + list(envs[j].step(actions[j]))
        states = torch.from_numpy(np.array([rollouts[i+num_steps*j][2] for j in range(len(actions))])).float().to(device)
    rollouts[-1] = len(envs)
    print(rollouts)
    print(states)
    return rollouts, states

# Using the rollouts returned by collect_rollouts function, updates the actor
# and critic models. You will need to:
# 1a. Compute targets for the critic using the current critic in model.
# 1b. Compute loss for the critic, and optimize it.
# 2a. Compute returns, or estimate for returns, or advantages for updating the actor.
# 2b. Set up the appropriate loss function for actor, and optimize it.
# Function can return actor and critic loss, for plotting.
def update_model(model, gamma, optim, rollouts, device, iteration, writer, use_advantage):
    # TODO
    actor_loss, critic_loss = 0., 0.
    episodes = rollouts[-1]
    rollouts = rollouts[:-1]
    batch_size = len(rollouts)//episodes
    size = len(rollouts)
    # print(rollouts[0][0].shape)
    s_all = torch.zeros((size, *rollouts[0][0].shape)).to(device)
    a_all = np.zeros((size, rollouts[0][1].shape[0]))
    s_prime_all = np.zeros((size, *rollouts[0][2].shape))
    r_all = torch.zeros(size).to(device)
    done_all = torch.zeros(size).to(device)
    for k in range(len(rollouts)):
        s_all[k, :], a_all[k, :], s_prime_all[k,:], r_all[k], done_all[k], _ = rollouts[k]
    all_rewards = []
    for episode in range(episodes):
        indices = slice(episode*batch_size, (episode + 1)*batch_size)
        done = done_all[indices]
        for i in reversed(range(batch_size)):
            if not done[i]:
                break
        s = s_all[indices][:i+1]
        a = a_all[indices][:i+1]
        s_prime = s_prime_all[indices][:i+1]
        actor, critic = model(s)
        policy_dists = model.actor_to_distribution(actor).probs.squeeze(1)
        log_prob = torch.log(torch.take_along_dim(policy_dists, torch.from_numpy(a).long().to(device), dim=1))
        s_prime = s_prime_all[indices][:i+1]
        r = r_all[indices][:i+1]

        all_rewards.append((1 - done[:i+1])*r[:i+1].sum())
        _, qval = model(torch.from_numpy(s_prime[-1:]).float().to(device))

        qvals = torch.zeros(len(r)).to(device)
        for t in reversed(range(len(r))):
            qval = r[t] + gamma*qval
            qvals[t] = qval
        qvals = qvals.detach()
        # print(indices, critic.size(), qvals.size(), critic[indices].size(), critic[indices])
        advantage = qvals - critic.view(-1) if use_advantage else critic.view(-1)
        actor_loss_t = (-log_prob*advantage.detach()).mean()
        # critic_loss_t = 0.5*(advantage**2).mean()
        critic_loss_t = torch.nn.functional.mse_loss(critic.view(-1), qvals)
        ac_loss = actor_loss_t + critic_loss_t
        actor_loss += actor_loss_t.item()
        critic_loss += critic_loss_t.item()
        optim.zero_grad()
        ac_loss.backward()
        # actor_loss_t.backward(retain_graph=True)
        # optim.step()
        # optim.zero_grad()
        # critic_loss_t.backward()
        optim.step()
    actor_loss /= episodes
    critic_loss /= episodes
    return actor_loss, critic_loss


def train_model_ac(model, envs, gamma, device, logdir, val_fn, advantage):
    model.to(device)
    train_writer = SummaryWriter(logdir / 'train')
    val_writer = SummaryWriter(logdir / 'val')

    # You may want to setup an optimizer, loss functions for training.
    optim = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    # TODO

    # Resetting all environments to initialize the state.
    num_steps, total_samples = 0, 0
    states = torch.from_numpy(np.array([e.reset() for e in envs])).float().to(device)

    for updates_i in range(num_updates):

        # Put model in training mode.
        model.train()

        # Collect rollouts using the policy.
        rollouts, states = collect_rollouts(model, envs, states, num_steps_per_rollout, device)
        num_steps += num_steps_per_rollout
        total_samples += num_steps_per_rollout*len(envs)


        # Use rollouts to update the policy and take gradient steps.
        actor_loss, critic_loss = update_model(model, gamma, optim, rollouts,
                                               device, updates_i, train_writer, advantage)
        log(train_writer, updates_i, 'train-samples', total_samples, 100, 10)
        log(train_writer, updates_i, 'train-actor_loss', actor_loss, 100, 10)
        log(train_writer, updates_i, 'train-critic_loss', critic_loss, 100, 10)
        log(train_writer, updates_i, None, None, 100, 10)


        # We are solving a continuing MDP which never returns a done signal. We
        # are going to manully reset the environment every few time steps. To
        # track progress on the training envirnments you can maintain the
        # returns on the training environments, and log or print it out when
        # you reset the environments.
        if num_steps >= reset_every:
            states = torch.from_numpy(np.array([e.reset() for e in envs])).float().to(device)
            num_steps = 0

        # Every once in a while run the policy on the environment in the
        # validation set. We will use this to plot the learning curve as a
        # function of the number of samples.
        cross_boundary = total_samples // val_every > \
            (total_samples - len(envs)*num_steps_per_rollout) // val_every
        if cross_boundary:
            model.eval()
            mean_reward = val_fn(model, device)
            log(val_writer, total_samples, 'val-mean_reward', mean_reward, 1, 1)
            log(val_writer, total_samples, None, None, 1, 1)
            model.train()
