# -*- coding: utf-8 -*-
# @Time    : 2021/2/24 11:12 上午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: utils.py
import numpy as np
import struct
import torch
import random
from redis import Redis
from typing import List, Tuple, Union


def mul(x):
    o = 1
    for i in x:
        o *= i
    return o


def random_choice_prob_index(p, axis=1):
    """
        this subtle function from https://github.com/openai/random-network-distillation/blob/master/ppo_agent.py
    """
    r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
    return (p.cumsum(axis=axis) > r).argmax(axis=axis)


# todo : change to mutil agents
def select_action(predictions: np.ndarray, **kwargs) -> np.ndarray:
    """
        Parameters
        ----------
        predictions.shape: [agent_num, action_num]

        Returns
        -------
        actions_oldlogps.shape: [agent_num, 2]
    """
    actions_oldlogps = np.zeros((predictions.shape[0], 2), dtype=kwargs['module_dtype'])
    if not kwargs['PLAY_MODE']:
        tmp_action = random_choice_prob_index(predictions)
        actions_oldlogps[:, 0] = tmp_action
        actions_oldlogps[:, 1] = np.log(predictions[range(predictions.shape[0]), tmp_action])
    else:
        actions_oldlogps[:, 0] = np.argmax(predictions, axis=1)

    return actions_oldlogps


if __name__ == "__main__":
    a = np.array([[1, 0, 0],
                  [0.1, 0.6, 0.3],
                  [0.3, 0.3, 0.4]
                  ])
    print(select_action(a, PLAY_MODE=False,
                        module_dtype=np.float32))