from pathlib import Path
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from absl import flags, app
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
FLAGS = flags.FLAGS

flags.DEFINE_spaceseplist('logdirs', [],
    'Space separated list of directories to plot results from.')
flags.DEFINE_string('output_file_name', 'out.pdf',
    'Output file to generate plot.')
flags.DEFINE_integer('seeds', 5,
    'Number of seeds per run')

def main(_):
    sns.color_palette()
    fig = plt.figure(figsize=(8,4))
    ax = fig.gca()
    print(FLAGS.logdirs)
    for logdir in FLAGS.logdirs:
        print(logdir)
        samples = []
        rewards = []
        for seed in range(FLAGS.seeds):
            logdir_ = Path(logdir) / f'seed{seed}'
            logdir_ = logdir_ / 'val'
            event_acc = EventAccumulator(str(logdir_))
            event_acc.Reload()
            _, step_nums, vals = zip(*event_acc.Scalars('val-mean_reward'))
            samples.append(step_nums)
            rewards.append(vals)
        samples = np.array(samples)
        assert(np.all(samples == samples[:1,:]))
        rewards = np.array(rewards)
        mean_rewards = np.mean(rewards, 0)
        std_rewards = np.std(rewards, 0)
        ax.plot(samples[0,:], mean_rewards, label=logdir)
        ax.fill_between(samples[0,:],
                        mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2)

    ax.legend(loc=4)
    ax.set_ylim([0, 210])
    ax.grid('major')
    fig.savefig(FLAGS.output_file_name + '.png', bbox_inches='tight')

    plt.clf()
    fig = plt.figure(figsize=(8,4))
    ax = fig.gca()
    for logdir in FLAGS.logdirs:
        print(logdir)
        samples = []
        prefix = logdir.split('_')[0]
        isdqn = (prefix == 'DQN')
        if isdqn:
            errors = []
        else:
            actor_loss = []
            critic_loss = []
        for seed in range(FLAGS.seeds):
            logdir_ = Path(logdir) / f'seed{seed}'
            logdir_ = logdir_ / 'train'
            event_acc = EventAccumulator(str(logdir_))
            event_acc.Reload()
            if isdqn:
                _, step_nums, loss = zip(*event_acc.Scalars('train-bellman-error'))
                errors.append(loss)
            else:
                _, step_nums, a_loss = zip(*event_acc.Scalars('train-actor_loss'))
                _, _, c_loss = zip(*event_acc.Scalars('train-critic_loss'))
                actor_loss.append(a_loss)
                critic_loss.append(c_loss)
            samples.append(step_nums)
        samples = np.array(samples)
        assert(np.all(samples == samples[:1,:]))
        if isdqn:
            errors = np.array(errors)
            mean_errors = np.mean(errors, 0)
            std_errors = np.std(errors, 0)
            ax.plot(samples[0,:], mean_errors, label=logdir)
            ax.fill_between(samples[0,:],
                            mean_errors-std_errors, mean_errors+std_errors, alpha=0.2)
        else:
            actor_loss = np.array(actor_loss)
            critic_loss = np.array(critic_loss)
            mean_actor_loss = np.mean(actor_loss, 0)
            std_actor_loss = np.std(actor_loss, 0)
            mean_critic_loss = np.mean(critic_loss, 0)
            std_critic_loss = np.std(critic_loss, 0)
            ax.plot(samples[0,:], mean_actor_loss, label=logdir + '_actor_loss')
            ax.fill_between(samples[0,:],
                            mean_actor_loss-std_actor_loss, mean_actor_loss+std_actor_loss, alpha=0.2)
            ax.plot(samples[0,:], mean_critic_loss, label=logdir + '_critic_loss')
            ax.fill_between(samples[0,:],
                            mean_critic_loss-std_critic_loss, mean_critic_loss+std_critic_loss, alpha=0.2)

    ax.legend(loc=4)
    # ax.set_ylim([0, 210])
    ax.grid('major')
    fig.savefig(FLAGS.output_file_name + '_train_loss.png', bbox_inches='tight')


if __name__ == '__main__':
    app.run(main)
