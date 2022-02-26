# -*- coding: utf-8 -*-
# @Time    : 2021/5/31 9:44 下午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: RND.py

import numpy as np
import torch
import time
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from USTC_lab.nn import *
from USTC_lab.nn import mlp, merge
from USTC_lab.data import Experience


class RNDNet(Basenn):
    def __init__(self, **kwargs):
        config, config_nn = kwargs['config'], kwargs['config_nn']
        super(RNDNet, self).__init__(config, config_nn)
        self.fixed = kwargs['fixed']
        self.mlp_layer = mlp(kwargs['input_mlp'])
        self.pre = kwargs['pre']
        self.epochs = config_nn.RND_EPOCH

        if self.fixed:
            self.requires_grad_(False)

        self.init_weight()

        # random seed
        self.update_time = 0

        self.optim = torch.optim.Adam(self.parameters(), lr=config_nn.RND_LEARNING_RATE)

    def states_normalization(self, states):
        return torch.clip((states - torch.mean(states)) / torch.std(states), -5, 5)

    def forward(self, x):
        x = self.states_normalization(x[:, :1, :, :])
        x = self.pre(x)
        return self.mlp_layer(x)

    def learn(self, states, labels):
        for _ in range(self.epochs):
            start_time = time.time()
            loss = 0.5 * torch.mean((labels - self(states)) ** 2)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.update_time += 1
            yield {"RndBackUpTime": time.time() - start_time,
                   "RndLoss": loss.item(),
                   }, self.update_time, True


class RND(Basenn):
    def __init__(self,
                 pre_f: PreNet = None,
                 pre_p: PreNet = None,
                 rnd_critic: Critic = None,
                 config=None,
                 config_nn=None):
        super(RND, self).__init__(config, config_nn)
        # fixed net
        self.f_net = RNDNet(input_mlp=config.RND_MLP_F_LIST,
                            fixed=True,
                            pre=pre_f,
                            config=config,
                            config_nn=config_nn)
        # predictor_net
        self.p_net = RNDNet(input_mlp=config.RND_MLP_P_LIST,
                            fixed=False,
                            pre=pre_p,
                            config=config,
                            config_nn=config_nn)

        self.rnd_critic = rnd_critic

    def forward(self, x):
        # TODO detach?

        return torch.mean((self.f_net(x) - self.p_net(x)) ** 2, dim=1, keepdim=True)

    def states_normalization(self, states):
        return torch.clip( (states - torch.mean(states)) / torch.std(states), -5, 5)

    def learn(self, states):
        # our original states contains n frams stack, but rnd net only need 1 frame
        # states = self.states_normalization(states[:, :1, :, :])
        labels = self.f_net(states)
        for loss_item, update_time, last in self.p_net.learn(states, labels.detach()):
            yield loss_item, update_time, last



if __name__ == "__main__":
    # Config.TASK_NAME = ""
    pre = PreNet(1, 'cuda')
    pre2 = PreNet(1, 'cuda')
    rndd = RND(pre_f=pre,
               pre_p=pre2)
    x = torch.rand([16, 1, 84, 84], dtype=torch.float32)
    # rnd = RNDNet(fixed=True,
    #           input_mlp=[(64, 1, "sigmoid")],
    #              pre=pre)
    print(rndd(x))
    # while True:
    #     for i in rndd.learn(x):
    #         print(i[0])
    # for k,v in rndd.f_net.named_parameters():
    #     print(k)
    # rnd2 = RNDNet(fixed=False,
    #           input_mlp=[(64, 1, "sigmoid")],
    #              pre=pre)
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # for k,v in rndd.p_net.named_parameters():
    #     print(k)