# -*- coding: utf-8 -*-
# @Time    : 2021/2/22 10:10 下午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: experience.py
#
import numpy as np
import struct
import math
import torch
import redis

from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Generator



def mul(x):
    out = 1
    for i in x:
        out *= i
    return out


class Experience:
    """
        mutil agents' exp in one env
    """
    __slots__ = ['states', 'actions', 'old_logps', 'rewards',
                 'dones', 'advs', 'values','durations','is_clean']

    def __init__(self, states,
                 advs=None,
                 actions=None,
                 old_logps=None,
                 values=None,
                 rewards=None,
                 dones=None,
                 durations=None,
                 is_clean=None):
        self.states = states
        self.actions = actions
        self.durations = durations
        self.old_logps = old_logps
        self.rewards = rewards
        self.dones = dones
        self.advs = advs            # advantage for ppo
        self.values = values
        # print("a",clean)
        self.is_clean = is_clean

    def __len__(self):
        return len(self.states[0])

    def get_xrapv(self):
        return [self.states, self.advs, self.actions, self.old_logps, self.values]

    def to_tensor(self, dtype=torch.float32, device='cpu'):
        for i in range(len(self.states)):
            self.states[i] = torch.tensor(self.states[i], dtype=dtype, device=device)
        self.advs = torch.tensor(self.advs, dtype=dtype, device=device)
        self.actions = torch.tensor(self.actions, dtype=dtype, device=device)
        self.old_logps = torch.tensor(self.old_logps, dtype=dtype, device=device)
        self.values = torch.tensor(self.values, dtype=dtype, device=device)

    def concatenate(self, data: "Experience"):
        if not data:
            return
        self.states = np.concatenate((self.states, data.states), axis=0)
        self.advs = np.concatenate((self.advs, data.advs), axis=0)
        self.actions = np.concatenate((self.actions, data.actions), axis=0)
        self.old_logps = np.concatenate((self.old_logps, data.old_logps), axis=0)
        self.values = np.concatenate((self.values, data.values), axis=0)

    def split(self, batch_size: int) -> Generator["Experience", None, None]:

        split_size = math.ceil(len(self.states) / batch_size)
        split_array = [i * batch_size for i in range(1, split_size)]

        splited_xrapv = []
        for item in self.get_xrapv():
            splited_xrapv.append(np.split(item, split_array, axis=0))
        for i in range(split_size):
            tmp = []
            for t in splited_xrapv:
                tmp.append(t[i])
            yield Experience(*tmp)

    @classmethod
    def batch_data_gene(cls, exps: List["Experience"]) -> Generator["Experience", None, None]:
        """batch all of agents' data
        following 64, is for cache, if not, to large numpy data will in error.
        set 128 258 also ok.
        """
        length = len(exps)
        index = 0
        while index < length:
            bexp = Experience.batch_data(exps=exps[index: index+64], clean=False)
            index += 64
            yield bexp

    @classmethod
    def concat_state(cls, states: List[List[np.ndarray]]) -> List[np.ndarray]:
        list_np_states = []
        
        for i in range(len(states[0])):
            xxx = []
            for j in range(len(states)):
                xxx.append(states[j][i])

            list_np_states.append(np.concatenate(xxx, axis=0))
        return list_np_states

    @classmethod
    def index_state(cls, states: List[np.ndarray], index: np.ndarray) -> List[np.ndarray]:
        return [each_state[index] for each_state in states]

    @classmethod
    def batch_data(cls, exps: List["Experience"], clean: bool = True) -> "Experience":
        '''
            if you concatenate step by step, it's very slow.
            batch all of training data at once
        '''
        states, advs, actions, old_logps, values = Experience.concat_state([exp.states for exp in exps ]), \
                                                   np.concatenate([exp.advs for exp in exps ],
                                                                  axis=0), \
                                                   np.concatenate([exp.actions for exp in exps ],
                                                                  axis=0), \
                                                   np.concatenate([exp.old_logps for exp in exps ],
                                                                  axis=0), \
                                                   np.concatenate([exp.values for exp in exps],
                                                                  axis=1)
        if clean:
            return Experience(states=states,
                              advs=advs,
                              actions=actions,
                              old_logps=old_logps,
                              values=values,
                              )
        else:
            # in multi agent ,check useful exp
            index = np.concatenate([exp.is_clean for exp in exps], axis=0)
            assert index.dtype == np.bool
            return Experience(states=Experience.index_state(states, index),
                                advs=advs[index],
                                actions=actions[index],
                                old_logps=old_logps[index],
                                values=values[:,index],
                                is_clean=np.array([True]*np.sum(index))
                             )
        
