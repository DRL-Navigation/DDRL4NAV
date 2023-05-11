# -*- coding: utf-8 -*-
# @Time    : 2021/2/23 4:10 下午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: __init__.py

# network
from USTC_lab.nn.utils import *


from USTC_lab.nn.base import Basenn, PreNet
from USTC_lab.nn.critic import Critic
from USTC_lab.nn.actor import CategoricalActor, Actor, GaussionActor
from USTC_lab.nn.RND import RND
from USTC_lab.nn.ppo import PPO
from USTC_lab.nn.GAIL import GAIL, Discriminator
from USTC_lab.nn.nav_encoder import NavPreNet, NavPedPreNet, NavPreNet1D
from USTC_lab.nn.atari_encoder import AtariPreNet
from USTC_lab.nn.mlp_encoder import MLPPreNet
NETWORK_MAP = {
    "ppo" : PPO,
    "gail": GAIL,
}
__all__ = [
    'PPO',
    'NavPreNet',
    'Basenn',
    "PreNet",
    'NETWORK_MAP',
    "CategoricalActor",
    "GaussionActor",
    "Critic",
    "AtariPreNet",
    "MLPPreNet",
    "Actor",
    'GAIL',
    "Discriminator",
    "RND",
    "mlp",
    "merge",
    "NavPedPreNet",
    "NavPreNet1D",

]