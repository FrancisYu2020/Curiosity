import numpy as np
import time

import matplotlib.pyplot as plt

class Logger():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        
        with open(self.log_dir, "w") as f:
            f.write(f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}")
            f.write(f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}")
            f.write(f"{'TimeDelta':>15}\n")
        
        self.reward_plot = log_dir / "rewards.jpg"
        self.episodes_plot = log_dir / "episodes.jpg"
        self.loss_plot = log_dir / "average-loss.jpg"
        self.q_plot = log_dir / "q-values.jpg"
        
        self.rewards = []
        self.episodes = []
        self.average_q_values = []
        self.average_loss = []

        self.moving_rewards = []
        self.moving_episodes = []
        self.moving_q_values = []
        self.moving_loss = []

        self.init_episode()

        self.time = time.time()

    def log(self, reward, q, loss):
        self.current_reward += reward
        self.current_episode += 1
        if loss:
            self.current_q += q
            self.current_loss_ct += 1
            self.current_loss += loss

    def log_episode(self):
        self.rewards.append(self.current_reward)
        self.episodes.append(self.current_episode)

        if(self.current_loss_ct > 0):
            self.average_loss.append(np.round(self.current_loss / self.current_loss_ct))
            self.average_q_values.append(np.round(self.current_q / self.current_loss_ct))
        else:
            self.average_loss.append(0)
            self.average_q_values.append(0)

        self.init_episode()

    def init_episode(self):
        self.current_reward = 0
        self.current_episode = 0
        self.current_loss = 0
        self.current_loss_ct = 0
        self.current_q = 0

    def record(self, episode, epsilon, step):
        mean_reward = np.round(np.mean(self.rewards[-100:]), 4)
        mean_length = np.round(np.mean(self.episodes[-100:]), 4)
        mean_loss = np.round(np.mean(self.average_loss[-100:]), 4)
        mean_q_value = np.round(np.mean(self.average_q_values[-100:]), 4)

        self.moving_rewards.append(mean_reward)
        self.moving_episodes.append(mean_length)
        self.moving_loss.append(mean_loss)
        self.moving_q_values.append(mean_q_value)

        previous = self.time
        self.time = time.time()

        delta_time = self.time - previous

        print("Episode: " + str(episode) + "| Step: " + str(step) + "| Epsilon: " + str(epsilon))
        print("Average Reward: " + str(mean_reward) + "| Avarege Length: " + str(mean_length) + "| Average Loss: " + str(mean_loss) + "| Average Q Value: " + str(mean_q_value))
        print("Time Elapsed: " + str(delta_time) + "| Current Time: " + str(self.time))

        with open(self.log_dir, "a") as f:
            f.write(f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_reward:15.3f}{mean_length:15.3f}{mean_loss:15.3f}{mean_q_value:15.3f}"
                f"{delta_time:15.3f}\n")
        
        for log_entry in ["moving_rewards", "moving_episodes", "moving_loss", "moving_q_values"]:
            plt.plot(getattr(self, f"{log_entry}"))
            plt.savefig(getattr(self, f"{log_entry}-plot"))
            plt.clf()