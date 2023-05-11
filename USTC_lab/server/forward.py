# -*- coding: utf-8 -*-
# @Time    : 2021/2/23 5:21 下午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: forward.py
import redis, time
import torch.nn.functional as F
import torch
import numpy as np

from threading import Thread
from multiprocessing import Process, Queue, Value
from typing import List


from USTC_lab.nn import Basenn, GaussionActor, CategoricalActor
from USTC_lab.data import LoggerFactory, EasyBytes
from USTC_lab.nn import GAIL


class ForwardThread(Thread):
    def __init__(self, net: Basenn,
                 predictor_id: int,
                 logger: LoggerFactory,
                 easy_bytes: EasyBytes,
                 exit_flag: Value,
                 configs
                 ):
        super(ForwardThread, self).__init__()
        self.setDaemon(True)
        self.net = net

        config, config_nn, config_env = configs['config'], configs['config_nn'], configs['config_env']

        self.predictor_id = predictor_id
        self.easy_bytes = easy_bytes
        self.logger_f = logger
        self.logger_f.update_tensor_tags('predict/', predictor_id)

        self.conn_pre = redis.Redis(config.PREDICTOR_REDIS_HOST,
                                    config.PREDICTOR_REDIS_PORT)
        self.pipe_pre = self.conn_pre.pipeline()
        self.conn_middle = redis.Redis(config.MIDDLE_REDIS_HOST,
                                       config.MIDDLE_REDIS_PORT)
        self.pipe_middle = self.conn_middle.pipeline()

        self.exit_flag = exit_flag

        self.tensortype = config_nn.MODULE_TENSOR_DTYPE
        self.nptype = config_nn.MODULE_NUMPY_DTYPE
        self.device = config_nn.DEVICE

        self.forward_states_key = config.TASK_NAME + config.PREDICTING_STATES_KEY
        self.pre_actionkey = config.TASK_NAME + config.PRE_ACTIONS_KEY
        self.update_tag = config.TASK_NAME + config.UPDATE_TAG_KEY

        self.env_dict_key = config.TASK_NAME + config.ENV_NUM_DICT_KEY
        env_dict = self.conn_pre.hgetall(self.env_dict_key)
        self.env_dict = {}
        for k, v in env_dict.items():
            self.env_dict[str(k)] = int(v)

        self.episode = 0
        self.timeout = config.TIME_OUT

        self.action_dim = config_nn.ACTIONS_DIM

        self.play_mode = config.PLAY_MODE or config.DEMONSTRATE_MODE

        self.config = config
        self.config_nn = config_nn

        self.sync = config.SYNC
        self.train_lock_key = config.TASK_NAME + config.TRAIN_LOCK_KEY

        self.batch_num_per_env = config_env['batch_num_per_env']
        self.agent_num_per_env = config_env['agent_num_per_env']

    def check_demonstrate(self) -> bool:
        """
            if demonstrate_mode == True, it means that this process only demonstrate state, action pairs
        """
        if self.play_mode:

            self.net.updatenn(path=self.config.DEMONSTRATE_LOAD_PATH,
                              conn=self.conn_middle)
            return True
        return False

    # def update_envid(self, env_id,):
    #     if not self.env_dict.get(env_id, None):
    #         tmp_env_dict = self.conn_pre.hgetall(self.env_dict_key)
    #         for k, v in tmp_env_dict.items():
    #             self.env_dict[str(k)] = int(v)

    # def get_env_batch_num(self, env_ids: List[str]):
    #     list_env_batch_num = []
    #     for env_id in env_ids:
    #         self.update_envid(env_id)
    #         list_env_batch_num.append(self.env_dict[env_id])
    #     return list_env_batch_num

    def state2tensor(self, states):
        for i in range(len(states)):
            states[i] = torch.tensor(states[i], dtype=self.tensortype, device=self.device)

    # TODO mutil thread unsafe now
    def run(self):
        pre_update = 0
        # if mode is demonstrate, it means we do not need to open training process.
        if not self.check_demonstrate():
            train_update = self.conn_middle.get(self.update_tag)
            while not train_update:
                time.sleep(0.3)
                train_update = self.conn_middle.get(self.update_tag)

        while self.exit_flag.value == b'0':
            byte_states = self.conn_pre.blpop(self.forward_states_key, timeout=self.timeout * 10)[1]    # get obs from redis
            while self.sync and int(self.conn_middle.get(self.train_lock_key)) == 1:                    # SYNC MODE
                time.sleep(0.1)
            start_time = time.time()
            if not self.play_mode:
                train_update = int(self.conn_middle.get(self.update_tag))

                if train_update > pre_update:
                    self.net.updatenn_by_redis(self.conn_middle)
                    pre_update = train_update

            env_ids, batch_states = self.easy_bytes.decode_forward_states(byte_states)
            self.state2tensor(batch_states)
            # print(batch_states, flush=True)
            # values : List[ [v1,v1, ...] , [v2,v2, ...] ]
            pi,  values = self.net(batch_states, play_mode=self.play_mode)
            distribution, _ = pi
            # actions : [a1, a2, a3, ....]
            if not self.play_mode:
                # actions : [a1, a2, a3, ....]
                actions = distribution.sample().to(self.tensortype)
                logps = self.net.actor.log_prob_from_distribution(distribution, actions)
            else:
                if isinstance(self.net.actor, GaussionActor):
                    actions = distribution
                elif isinstance(self.net.actor, CategoricalActor):
                    actions = torch.argmax(distribution, dim=1).to(self.tensortype)
                logps = np.zeros([distribution.shape[0]], dtype=self.nptype)
            # values :  [[v1,v1, ...] , [v2,v2, ...]]
            values = torch.stack(values, dim=0)
            # if not self.net.soft_max_grid:
            #     actions = F.softmax(actions, dim=-1)
            list_forward_np_data = [actions, logps, values]
            # check RND
            if self.net.rnd:
                # original net output ：intrinsic_rewards: [[r], [r], ...]
                # after [: ,0]
                # intrinsic_rewards: [r, r, ...]
                intrinsic_rewards = self.net.get_rnd(batch_states)[:, 0]
                list_forward_np_data.append(intrinsic_rewards)

            # check GAIL
            if isinstance(self.net, GAIL):
                # D_rewards: [[r], [r], ...]
                D_rewards = self.net( (batch_states, actions.reshape(actions.shape[0], self.action_dim) ) )
                # D_rewards: [r, r, ...]
                D_rewards = self.config_nn.D_REWARD_FUNCTION(D_rewards)[:, 0]
                list_forward_np_data.append(D_rewards)

            # action
            list_forward_np_data[0] = list_forward_np_data[0].cpu().detach().numpy().astype(self.nptype)
            # each env process may consists of different numbers of robots
            list_env_batch_num: List[int] = [self.batch_num_per_env * self.agent_num_per_env] * len(env_ids)
            list_bytes_data: List[bytes] = self.easy_bytes.encode_forward_return_data(list_forward_np_data, list_env_batch_num)

            # set actions to redis
            i = 0
            for byte_env_data in list_bytes_data:
                self.pipe_pre.lpush(self.pre_actionkey.format(env_ids[i]), byte_env_data)
                i += 1
            self.pipe_pre.execute()

            # ms
            once_time = (time.time() - start_time) * 1000
            self.logger_f.add((once_time, self.episode), "ForwardTime-ms")
            self.episode += 1
        print("forward exit !", flush=True)