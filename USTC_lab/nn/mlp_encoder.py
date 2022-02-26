"""
mlp dealing module
serving for some classical or mujoco game.
"""
import torch
from torch import nn
import torch.nn.functional as F

from USTC_lab.nn import PreNet
from USTC_lab.nn import mlp

class MLPPreNet(PreNet):
    def __init__(self,
                 input_dim=4,
                 last_output_dim=128,
                 ):
        super(MLPPreNet, self).__init__()
        self.fc0 = mlp([ (input_dim, last_output_dim, "relu") ])

        # assert self.fc0.out_features == last_output_dim


    def forward(self, state):
        # encoded_image = self._encode_image(state[0])
        # x = self.fc0(encoded_image)
        # x = torch.cat((x, state[1]), dim=1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        return self.fc0(state[0])

