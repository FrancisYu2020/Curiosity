import torch
import torch.nn as nn
import numpy as np

#TODO: This implementation follows the original implemetation and hence the action is an one-hot vector. Instead, we may want to further do action embedding to make it potentially easier to learn as an extension for this project
# intrinsic curiosity module
'''
The intrinsic curiosity module consists of the forward and the inverse model. The inverse model first maps the input state
(st) into a feature vector φ(st) using a series of four convolution layers, each with 32 filters, kernel size 3x3, stride
of 2 and padding of 1. ELU non-linearity is used after each convolution layer. The dimensionality of φ(st) (i.e.
the output of the fourth convolution layer) is 288. For the inverse model, φ(st) and φ(st+1) are concatenated into a
single feature vector and passed as inputs into a fully connected layer of 256 units followed by an output fully connected layer with 4 units to predict one of the four possible
actions. 
'''
class InverseModel(nn.Module):
    '''
    The inverse model by default will transform the resized grey-scale image batch to a fixed dim feature vector [None, 1, 42, 42] -> [None, 288]
    '''
    def __init__(self, image_size=42, nConvs=4, action_dim=4) -> None:
        super().__init__()
        module_list = [nn.Conv2d(1, 32, 3, 2), nn.ELU()] + [nn.Conv2d(32, 32, 3, 2), nn.ELU()]*(nConvs - 1)
        self.feature_size = image_size
        for _ in range(nConvs):
            self.feature_size = (self.feature_size - 2)//2 + 1
        self.feature_size = feature_size**2 * 32
        self.feature = nn.Sequential(*module_list)
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * self.feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, s_curr, s_next):
        phi_curr = self.feature(s_curr).flatten(start_dim=1)
        phi_next = self.feature(s_next).flatten(start_dim=1)
        feature = torch.cat([phi_curr, phi_next], dim=1)
        a_pred = self.inverse_model(feature)
        return phi_curr, phi_next, a_pred

class ForwardModel(nn.Module):
    '''
    The forward model is constructed by concatenating φ(st) with at and passing it into a sequence of two fully connected layers with 256 and 288 units respectively.
    '''
    def __init__(self, feature_size=288, action_dim=4) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_size + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_size)
        )

    def forward(self, phi_curr, action):
        feature = torch.cat([action, phi_curr], dim=1)
        return self.forward_model(feature)

class ICM(nn.Module):
    '''
    The value of loss weight in the original paper follows that β is 0.2, and λ is 0.1. The Equation (7) is minimized with learning rate of 1e − 3.
    '''
    def __init__(self, image_size=42, nConvs=4, action_dim=4, beta=0.2):
        super().__init__()
        self.inverse_model = InverseModel(image_size, nConvs, action_dim)
        self.forward_model = ForwardModel(self.inverse_model.feature_size, action_dim)
        self.action_criterion = nn.CrossEntropyLoss()
        self.forward_criterion = nn.MSELoss()
        self.beta = beta
        #TODO: add action embedding and corresponding module to compute loss

    def forward(self, action, s_curr, s_next):
        phi_curr, phi_next, a_pred = self.inverse_model(s_curr, s_next)
        phi_next_pred = self.forward_model(phi_curr, action)
        return phi_next_pred, phi_next, a_pred
    
    def intrinsic_loss(self, phi_next_pred, phi_next, a_pred, action, return_all=False):
        action_loss = self.action_criterion(a_pred, action)
        forward_loss = 0.5 * self.forward_criterion(phi_next_pred, phi_next)
        if return_all:
            return action_loss, forward_loss, (1 - self.beta) * action_loss + self.beta * forward_loss
        return (1 - self.beta) * action_loss + self.beta * forward_loss