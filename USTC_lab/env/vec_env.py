# -*- coding: utf-8 -*-
# @Time    : 2021/6/27 7:31
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: vec_env.py
import numpy as np
import time

from typing import List, Dict


class VecEnv:
    # TODO support complex states instances
    # TODO 需要用户自己填入 输入网络的数据 函数或者配置什么的。
    # TODO 我这边只需要实现通信就行，用户输入数据即可，在另一端取数据。
    def __init__(self, envs: List):
        self.envs = envs
        self.states = [None] * len(envs)
        self.rewards = [None] * len(envs)
        self.downs = [None] * len(envs)
        self.tmp_infos: List[dict] = [None] * len(envs)
        self.infos: Dict = {}

    def _concat_state(self) -> List[np.ndarray]:
        list_np_states = []
        for i in range(len(self.states[0])):
            xxx = []
            for j in range(len(self.states)):
                xxx.append(self.states[j][i])

            list_np_states.append(np.concatenate(xxx, axis=0))
        return list_np_states

    def _concat_info(self):# -> Dict[str: np.ndarray]:
        keys = self.tmp_infos[0].keys()
        for key in keys:
            self.infos[key] = np.concatenate([dict_info[key] for dict_info in self.tmp_infos], axis=0)

        return self.infos

    def step(self, actions: np.ndarray):
        i = 0
        for env in self.envs:
            s, r, d, info = env.step(actions[i])
            self.states[i] = s
            self.rewards[i] = r
            self.downs[i] = d
            self.tmp_infos[i] = info
            i += 1

        return self._concat_state(),\
               np.concatenate(self.rewards, axis=0),\
               np.concatenate(self.downs, axis=0),\
               self._concat_info()

    def reset(self, **kwargs):
        self.states = [env.reset(**kwargs) for env in self.envs]
        return self._concat_state()

    def render(self, *args):
        # for env in self.envs:
        #   env.render()
        self.envs[0].render()

    def seed(self, seeds=None):
        if seeds is None:
            s = int(time.time() % 1 * 1000)
            seeds = range(s, s+len(self.envs))
        elif isinstance(seeds, int):
            s = seeds % 67
            seeds = range(s, s+len(self.envs))
        [env.seed(seed) for env, seed in zip(self.envs, seeds)]

    @property
    def done(self):
        for env in self.envs:
            if not env.done:
                return False
        return True

    def __len__(self):
        return len(self.envs)



def make_vecenv(envs):
    return VecEnv(envs)
