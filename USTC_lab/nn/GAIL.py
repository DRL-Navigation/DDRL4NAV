# -*- coding: utf-8 -*-
# @Time    : 2021/2/23 4:11 下午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: ppo.py
import numpy as np
import torch
import time
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Generator, Tuple, Dict, Union

from USTC_lab.nn import *
from USTC_lab.nn import mlp, merge
from USTC_lab.data import Experience
from USTC_lab.data import MimicExpFactory, MimicExpReader
from torch.utils.data import Dataset, DataLoader


class Discriminator(Basenn):
    def __init__(self, **kwargs):
        config, config_nn = kwargs['config'], kwargs['config_nn']
        super(Discriminator, self).__init__(config, config_nn)
        input_mlp = config.GAN_D_MLP_LIST
        pre = kwargs['pre']

        self.mlp_layer = mlp(input_mlp)
        self.pre = pre
        self.device = config.DEVICE
        self.dtype = config_nn.MODULE_TENSOR_DTYPE

        # WGAN recommend RMSprop rather than Adam or momentum
        self.optim = torch.optim.RMSprop(self.parameters(), lr=config_nn.GAN_D_LEARNING_RATE, alpha=0.9)
        self.optim_decay = torch.optim.lr_scheduler.StepLR(self.optim, step_size=250, gamma=0.95)
        # self.optim_decay = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.99)
        # self.optim_decay = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[500, 700, 800, 900], gamma=0.9)
        self.epochs = config_nn.GAN_D_EPOCH
        self.accumulation_steps = config_nn.GRAD_ACCUMULATION_STEP

        self.WGAN_clip_grad_num = config_nn.WGAN_CLIP_GRAD_NUM

        self.update_time = 0

        self.expert_data: DataLoader = self._get_data(config.MIMIC_START_LOAD_PATH, config.TASK_TYPE, config_nn.GAN_D_BATCH_SIZE)
        self.config = config
        self.config_nn = config_nn

        self.action_dim = config.ACTIONS_DIM

    def _get_data(self, data_path, task_type, batch_size, ) -> DataLoader:
        # print(data_path, flush=True)
        dataset = MimicExpFactory().mimic_reader(task_type,
                                                 data_path,
                                                 self.dtype,
                                                 self.device)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        # x: (S, a)
        # output: one dim , means score given by discriminator
        state, action = x
        if self.pre:
            state = self.pre(state)

        xx = merge(state, action)
        return self.mlp_layer(xx)

    def learn(self, data: Experience):

        for epoch in range(1, self.epochs + 1):
            # TODO iter(expert_data)
            start_time = time.time()
            for expert_batch in self.expert_data:
                # print(expert_batch[1])
                print("get expert data costs: ", time.time() - start_time)
                # TODO try to add H(pi) ?
                g_loss = torch.mean(self( (data.states, data.actions.reshape(data.actions.shape[0], self.action_dim)) ))
                expert_loss = -torch.mean(self( (expert_batch[0].to(self.device).to(self.dtype), expert_batch[1].to(self.device).to(self.dtype)) ))
                print("data 2 GPU time:", time.time() - start_time)
                totalDloss = g_loss + expert_loss
                self.optim.zero_grad()
                totalDloss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.WGAN_clip_grad_num)
                self.optim.step()
                self.optim_decay.step()
                print("In GanD:",time.time() - start_time)
                self.update_time += 1
                yield {"Gail[D]BackUpTime": time.time() - start_time,
                       "Gail[D]Loss": totalDloss.item(),
                       }, self.update_time, True

                # print("[D] update time",self.update_time, flush=True)
                break

                #if epoch % self.accumulation_steps == 0:
            #    self.optim.step()
            #    self.optim.zero_grad()




class GAIL(Basenn):
    def __init__(self,
                 generator: Union[PPO],
                 discriminator: Discriminator,
                 gail_critic: Critic,
                 ):
        super(GAIL, self).__init__(discriminator.config, discriminator.config_nn)
        self.device = discriminator.config.DEVICE
        self.generator = generator
        self.discriminator = discriminator

        self.actor = generator.actor
        self.rnd = self.generator.rnd
        self.gail_critic = gail_critic
        self.generator.add_critic(self.gail_critic)
        self.generator.gail_critic = True

    def _train_generator(self, data: Experience) -> Generator[Tuple[Dict, int, bool], None, None]:
        return self.generator.learn(data)

    def _train_discriminator(self, data: Experience) -> Generator[Tuple[Dict, int, bool], None, None]:
        return self.discriminator.learn(data)

    # def _reward_reshaping(self, data: Experience):
    #     """
    #         use D(s,a) to give additional reward in TRPO|PPO
    #         i find two ways to reshape reward,
    #             * directly use f(D(s,a)) as reward
    #             * f(D(s,a)) as a additional reward
    #     """
    #     # the first way:
    #
    #
    #     # the second way:
    #     data.advs += self.discriminator((data.states, data.actions)) * self.discriminator_landa
    def get_rnd(self, states):
        return self.rnd(states)

    def forward(self, x, act=None, play_mode=False):
        # G
        if not isinstance(x, tuple):
            actions, values = self.generator(x, act, play_mode)
            return actions, values
        # D
        if isinstance(x, tuple) and len(x) == 2:
            D_reward = self.discriminator(x)
            return D_reward

    def learn(self, data: Experience):
        s = time.time()
        for loss_item, update_time, last in self._train_discriminator(data):
            # print("GAN D", loss_item, update_time, flush=True)
            yield loss_item, update_time, False
        print("GAN D time:",time.time() - s)
        s = time.time()
        for loss_item, update_time, last in self._train_generator(data):
            yield loss_item, update_time, True
        print("PPO  time", time.time() - s, flush=True  )



if __name__ == "__main__":
    from  USTC_lab.config import Config, ConfigNN
    import copy
    config = Config("Breakout-v4")
    config_nn = ConfigNN()
    config.TASK_NAME = "test"
    actor, critic, prenet = CategoricalActor(len(config.ACTIONS), device=config.DEVICE,
                                             soft_max_grid=config_nn.SOFT_MAX_GRID), Critic(
        device=config.DEVICE), PreNet(config.STACKED_FRAMES, device=config.DEVICE)
    rnd_critic = copy.deepcopy(critic) if config_nn.RND_VALUE_TRICK else None
    # RND input frame is 1 based on RND paper.
    rnd_net = RND(pre_f=PreNet(1, device=config.DEVICE),
                  pre_p=PreNet(1, device=config.DEVICE),
                  rnd_critic=rnd_critic,
                  config=config,
                  config_nn=config_nn)
    gail_critic = copy.deepcopy(critic)
    ppo_net = PPO(actor, critic, prenet, rnd_net, config, config_nn).to(config.DEVICE)
    D_net = Discriminator(pre=PreNet(config.STACKED_FRAMES, device=config.DEVICE),
                          config=config,
                          config_nn=config_nn).to(config.DEVICE)
    net = GAIL(generator=ppo_net, discriminator=D_net, gail_critic=gail_critic).to(config.DEVICE)
    d = {}
    for k, v in net.named_parameters():
        d[k] = v
        print(k)
    net.load_state_dict(d, strict=False)
    states = torch.zeros([4,4,84,84], device=config.DEVICE)
    actions = torch.ones([4,1], device=config.DEVICE)
    exp = Experience(states, actions=actions)
    while True:
        for _ in net.discriminator.learn(exp):
            pass
    # net = GAIL(generator=ppo_net, discriminator=D_net).to(config.DEVICE)
    # print(D_net( (torch.zeros([4,4,84,84], device=config.DEVICE), torch.ones([4,1], device=config.DEVICE))) )