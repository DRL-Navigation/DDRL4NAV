# -*- coding: utf-8 -*-
# @Time    : 2021/2/24 4:34 下午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: __init__.py
from USTC_lab.config.utils import *
from USTC_lab.config.config_nn import ConfigNN
from USTC_lab.config.base_config import BaseConfig

# f=open("ATARI_LIST.txt")
# l=[]
# for line in f.readlines()[3:]:
#     l.append(line.strip().split("-")[0])
# print(l)
__all__ = [
    "BaseConfig",
    "ConfigNN",
]