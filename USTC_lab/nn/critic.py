from torch import nn

from USTC_lab.nn import mlp


class Critic(nn.Module):

    def __init__(self, device='cpu', last_input_dim=512, pre=None):
        super(Critic, self).__init__()
        self.device = device
        self.critic_linear = nn.Linear(last_input_dim, 1)
        self.pre = pre

    def forward(self, x):
        # if not share cnn net
        if self.pre:
            x2 = self.pre(x)
        else:
            x2 = x
        # if share cnn net, go to mlp layer directly.

        return self.critic_linear(x2)#.squeeze()