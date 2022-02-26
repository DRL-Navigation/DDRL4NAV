# -*- coding: utf-8 -*-
# @Time    : 2021/2/23 4:11 下午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: ppo.py
import numpy as np
import torch
import time
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from typing import Union

from USTC_lab.nn import *
from USTC_lab.data import Experience


class PPO(Basenn):
    def __init__(self, 
                 actor, 
                 critic, 
                 prenet=None, 
                 rnd: RND=None,
                 config=None,
                 config_nn=None):
        super(PPO, self).__init__(config, config_nn)
        self.device = config.DEVICE
        self.prenet = prenet
        self.actor = actor
        self.critic = critic
        self._critics = [self.critic]
        self.rnd = rnd
        if rnd and rnd.rnd_critic:
            self._critics.append(rnd.rnd_critic)
        self.gail_critic = False

        self.share_cnn_net = config_nn.SHARE_CNN_NET
        # self.optim = torch.optim.Adam(
        #     set(actor.parameters()).union(critic.parameters()).union(prenet.parameters()), config_nn.LEARNING_RATE
        # )
        self.optim = torch.optim.Adam(self.parameters(), config_nn.LEARNING_RATE)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), config_nn.ACTOR_LEARNING_RATE)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), config_nn.CRITIC_LEARNING_RATE)
        self.clip_grad = config_nn.CLIP_GRID
        self.clip_grad_num = config_nn.CLIP_GRID_NUM

        self.v_loss_theta = config_nn.V_LOSS_THETA
        self.ent_loss_theta = config_nn.ENTROPY_LOSS_THETA
        self.ppo_clip = config_nn.PPO_CLIP
        self.duel_ppo_clip = config_nn.DUEL_PPO_CLIP

        self.training_iter_time = config_nn.TRAINING_ITER_TIME

        # define vloss function
        if config_nn.SMOOTH_L1_LOSS:
            self.vlossf = F.smooth_l1_loss
        else:
            self.vlossf = lambda x, y: torch.mean((x - y) ** 2) / 2

        self.update_time = 0

        # self.soft_max_grid = actor.soft_max_grid

    def add_critic(self, critic: Critic):
        self._critics.append(critic)

    def get_rnd(self, states):
        return self.rnd(states)

    def states_normalization(self, states):
        return states / 255

    def forward(self, states, act=None, play_mode=False):
        if self.prenet:
            states = self.prenet(states)
        return self.actor(states, act, play_mode), [critic(states) for critic in self._critics]

    def learn(self, data: Experience):
        torch.cuda.empty_cache()
        for _ in range(self.training_iter_time):
            start_time = time.time()
            # distribution: Union[Categorical, Normal]
            pi, values = self(data.states, data.actions)
            distribution, log_p = pi
            # TODO require_grid = False in begining
            ratio = torch.exp(log_p - data.old_logps)
            # duel ppo
            actor_loss = -torch.mean(torch.where(data.advs > 0, torch.min(ratio * data.advs,
                                               torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * data.advs),
                                     torch.max(torch.min(ratio * data.advs,
                                               torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * data.advs),
                                               self.duel_ppo_clip * data.advs)
                                     ))
            # suppose 3 critic net at most
            # TODO if there are more than 3 critic net, the following code should reconstruct
            ppov_loss = self.vlossf(data.values[0, :], values[0].squeeze())
            assert data.values[0, :].shape == values[0].squeeze().shape
            if self.rnd and self.rnd.rnd_critic:
                rndv_loss = self.vlossf(data.values[1, :], values[1].squeeze())
            else:
                rndv_loss = 0
            if self.gail_critic:
                gailv_loss = self.vlossf(data.values[-1, :], values[-1].squeeze())
            else:
                gailv_loss = 0

            entropy_loss = torch.mean(distribution.entropy())
            v_loss = ppov_loss + rndv_loss + gailv_loss
            total_loss = actor_loss + v_loss * self.v_loss_theta - entropy_loss * self.ent_loss_theta

            if self.share_cnn_net:
                self.optim.zero_grad()
                total_loss.backward()

                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_num)

                self.optim.step()
            else:
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()

                actor_loss.backward()
                v_loss.backward()

                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_num)

                self.actor_optim.step()
                self.critic_optim.step()

            self.update_time += 1
            loss_log = {
                "PpoTotalLoss": total_loss.item(),
                "ActorLoss": actor_loss.item(),
                "VLoss": v_loss.item(),
                "EntLoss": entropy_loss.item(),
                "PpoBackUpTime": time.time() - start_time}

            # if rndv_loss:
            #     loss_log["VRNDLoss"] = rndv_loss.item()

            yield loss_log, self.update_time, True
        # update rnd net
        if self.rnd:
            for loss, update_time, last in self.rnd.learn(data.states):
                yield loss, update_time, last


