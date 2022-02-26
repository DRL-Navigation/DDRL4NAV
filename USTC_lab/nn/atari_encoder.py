"""
cnn dealing module
"""
import torch
import torch.nn.functional as F

from torch import nn

from USTC_lab.nn import PreNet

class AtariPreNet(PreNet):
    def __init__(self, num_inputs=1,
                 last_output_dim=512,
                 device='cpu'):
        super(AtariPreNet, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.linear = nn.Linear(3136, 512)
        assert self.linear.out_features == last_output_dim

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x[0]), inplace=False)
        x = F.leaky_relu(self.conv2(x), inplace=False)
        x = F.leaky_relu(self.conv3(x), inplace=False)
        # x4 = F.relu(self.conv4(x3), inplace=False)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x







if __name__ == "__main__":
    net = AtariPreNet()
    opt = torch.optim.Adam(
        net.parameters(), 0.1
    )
    a1=net(torch.zeros([1,1,84,84],
                    dtype=torch.float32) )

    print(a1.squeeze().shape)
    opt.zero_grad()
    loss = torch.mean(a1.squeeze() - torch.zeros(512))
    loss.backward()
    opt.step()