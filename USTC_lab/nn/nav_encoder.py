"""
cnn dealing module
"""
import torch
from torch import nn
import torch.nn.functional as F

from USTC_lab.nn import PreNet
from USTC_lab.nn import mlp
import logging

class NavPreNet(PreNet):
    def __init__(self,
                 image_channel=1,
                 last_output_dim=512,
                 ):
        super(NavPreNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(image_channel, 64, 3, stride=1, padding=(1,1))
        self.conv2 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=(1,1))
        self.conv3 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1,1))
        self.fc0 = mlp([ (256 * 6 * 6, 512, "relu")])
        # self.fc1 = nn.Linear(512 + 9, 512)
        self.fc1 = mlp([ (512 + 9, 512, "relu")])
        self.fc2 = nn.Linear(512, 512)
        # self.fc3 = nn.Linear(512, 512)
        # assert self.fc2.out_features == last_output_dim

    def _encode_image(self, image):
        image_x = F.max_pool2d(F.relu(self.conv1(image)), 2, stride=2)
        image_x = F.max_pool2d(F.relu(self.conv2(image_x)), 2, stride=2)
        image_x = F.max_pool2d(F.relu(self.conv3(image_x)), 2, stride=2)
        image_x = image_x.view(image_x.size(0), -1)
        return image_x

    def forward(self, state):
        encoded_image = self._encode_image(state[0])
        x = self.fc0(encoded_image)
        x = torch.cat((x, state[1]), dim=1)
        # x = state[1]
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        return x

class NavPedPreNet(PreNet):
    def __init__(self,
                 image_channel=4,
                 last_output_dim=512,
                 ):
        super(NavPedPreNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(image_channel, 64, 3, stride=1, padding=(1,1))
        self.conv2 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=(1,1))
        self.conv3 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1,1))
        # self.fc0 = nn.Linear(256 * 6 * 6, 512)
        self.fc0 = mlp([ (256 * 6 * 6, 512, "relu")])
        # self.fc1 = nn.Linear(512 + 9, 512)
        self.fc1 = mlp([ (512 + 9, 512, "relu")])
        self.fc2 = nn.Linear(512, 512)

        # self.fc3 = nn.Linear(512, 512)
        # assert self.fc2.out_features == last_output_dim

    def _encode_image(self, image):
        image_x = F.max_pool2d(F.relu(self.conv1(image)), 2, stride=2)
        image_x = F.max_pool2d(F.relu(self.conv2(image_x)), 2, stride=2)
        image_x = F.max_pool2d(F.relu(self.conv3(image_x)), 2, stride=2)
        image_x = image_x.view(image_x.size(0), -1)
        return image_x

    def forward(self, state):
        encoded_image = self._encode_image(torch.cat([state[0], state[2]], axis=1))
        x = self.fc0(encoded_image)
        x = torch.cat((x, state[1]), dim=1)
        # x = state[1]
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        return x
if __name__ == "__main__":
    net = NavPreNet(1,9)
    opt = torch.optim.Adam(
        net.parameters(), 0.1
    )
    a1=net([torch.zeros([1,1,48,48],
                    dtype=torch.float32),
            torch.zeros([1,9],dtype=torch.float32)] )

    # print(a1.squeeze().shape)
    # opt.zero_grad()
    # loss = torch.mean(a1.squeeze() - torch.zeros(512))
    # loss.backward()
    # opt.step()