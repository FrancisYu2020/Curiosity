#Author Francis Yu

from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class convBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel=3, stride=2, padding=1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel, stride, padding),
            nn.ELU()
        )

    def forward(self, x):
        return self.conv(x)

class DQNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_layers, action_dim, device) -> None:
        super().__init__()
        self.device = device
        input_channel, input_h, _ = input_dim
        layers = [convBlock(input_channel, hidden_layers[0])]
        output_h = (input_h - 1)//2 + 1
        for i, _ in enumerate(hidden_layers[:-1]):
            layers.append(convBlock(hidden_layers[i], hidden_layers[i+1]))
            output_h = (output_h - 1)//2 + 1
        self.layers = nn.Sequential(*layers)
        self.q_value = nn.Sequential(
            nn.Linear(hidden_layers[-1] * output_h**2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    # def forward(self, curr_state):
    #     feature = self.layers(curr_state).flatten(start_dim=1)
    #     return self.q_value(feature)
    
    def forward(self, curr_state,mode=""):
        feature = self.layers(curr_state).flatten(start_dim=1)
        if (mode == "aux"):
            return feature
        return self.q_value(feature)

    def act(self, x, epsilon=0.):
        qvals = self.forward(x)
        act = torch.argmax(qvals, 1, keepdim=True)
        if epsilon > 0:
            act_random = torch.multinomial(torch.ones(qvals.shape[1],), 
                                           act.shape[0], replacement=True)
            act_random = act_random.reshape(-1,1).to(self.device)
            combine = torch.rand(qvals.shape[0], 1) > epsilon
            combine = combine.float().to(self.device)
            act = act * combine + (1-combine) * act_random
            act = act.long()
        return act
        

class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(ActorCriticPolicy, self).__init__()
        # self.device = device
        input_channel, input_h, _ = input_dim
        layers = [convBlock(input_channel, hidden_layers[0])]
        output_h = (input_h - 1)//2 + 1
        for i, _ in enumerate(hidden_layers[:-1]):
            layers.append(convBlock(hidden_layers[i], hidden_layers[i+1]))
            output_h = (output_h - 1)//2 + 1
        self.layers = nn.Sequential(*layers)
        self.actor = nn.Sequential(
            # nn.Linear(hidden_layers[-1] * output_h**2, output_dim),
            nn.Linear(hidden_layers[-1] * output_h**2, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_layers[-1] * output_h**2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # self.actor = nn.Linear(256, output_dim)
        # self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.layers(x).flatten(start_dim=1)
        # print(x.size())
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic

    def actor_to_distribution(self, actor):
        action_distribution = Categorical(logits=actor.unsqueeze(-2))
        return action_distribution

    def act(self, x, sample=False):
        actor, critic = self.forward(x)
        action_distribution = self.actor_to_distribution(actor)
        if sample:
            action = action_distribution.sample()
        else:
            action = action_distribution.probs.argmax(-1)
        return action
