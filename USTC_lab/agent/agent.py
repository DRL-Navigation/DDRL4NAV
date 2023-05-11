# -*- coding: utf-8 -*-
# @Time    : 2021/2/26 4:52 下午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: agent.py
import numpy as np
import time
import random
import logging

from multiprocessing import Value
from multiprocess.context import Process
from threading import Thread
from redis import Redis
from copy import deepcopy
from typing import *

# logging.basicConfig(level=logging.DEBUG)

from USTC_lab.data import Experience, LoggerFactory, MimicExpWriter, EasyBytes
from USTC_lab.agent import TrainingQueue, PredictingQueue, Status


# 每个执行环境对应一个Agents
class Agents(Process):
    def __init__(self,
                 process_env_id: str,
                 logger: LoggerFactory,
                 # env_f=None,
                 easy_bytes: EasyBytes = None,
                 vector_envs=None,
                 pre_queue: PredictingQueue = None,
                 train_queue: TrainingQueue = None,
                 exit_flag: Value = Value('c', b'0'),
                 mimic_w: MimicExpWriter = None,
                 config = None,
                 config_nn = None,
                 config_env = None,
                 ):
        self.config = config
        self.config_nn = config_nn
        self.config_env = config_env
        super(Agents, self).__init__()
        self.easy_bytes = easy_bytes
        self.conn = Redis(self.config.PREDICTOR_REDIS_HOST, self.config.PREDICTOR_REDIS_PORT)
        self.pipe = self.conn.pipeline()

        self.agent_num_per_env = self.config_env["agent_num_per_env"]            # mutliagent in one env
        self.batch_num_per_env = self.config_env['batch_num_per_env']        # batch env num
        self.str_process_env_id = process_env_id   # like "localhost_0"
        self.int_process_env_id = int(self.str_process_env_id.split("_")[-1])
        self.machine_ip = process_env_id.split("_")[0]

        self.all_num = self.agent_num_per_env * self.batch_num_per_env
        # register  ENV_NUM_DICT
        self.conn.hset(self.config.TASK_NAME + self.config.ENV_NUM_DICT_KEY, self.str_process_env_id, self.all_num)

        assert vector_envs is not None and len(vector_envs) > 0
        assert pre_queue is not None
        assert train_queue is not None

        self.env = vector_envs
        self.env.seed()

        self.action_dim = self.config_nn.ACTIONS_DIM
        self.action_key = self.config.TASK_NAME + self.config.PRE_ACTIONS_KEY.format(self.str_process_env_id)
        self.exit_flag: Value = exit_flag
        self.logger_f = logger
        self.logger_f.update_tensor_tags('env/', self.str_process_env_id)
        self.episode = 0
        self.render = config.RENDER

        self.pre_queue = pre_queue
        self.train_queue = train_queue
        
        self.network_type = self.config_nn.NETWORK_TYPE
        self.model_dtype = self.config_nn.MODULE_NUMPY_DTYPE
        self.model_dtype_bytes = self.config_nn.MODULE_BITS // 8

        self.discounts_tmp, self.landa = [self.config_nn.EXTRINSIC_DISCOUNT], self.config_nn.LANDA

        self.T = self.config.TIME_MAX
        self.r_logfre = self.config.LOG_REWARD_FREQUENCY

        # init status instance
        self.status: Status = Status(agent_num=self.all_num,
                                     model_dtype=self.model_dtype,
                                     logger_f=self.logger_f)

        self.mimic_w = mimic_w

        self.time_out = self.config.TIME_OUT
        self.rnd = self.config_nn.USE_RND
        self.value_dim_num = self.reward_dim_num = 1
        # values dim , if gail, values will be [v, d_r]
        self.network_type = 'ppo'
        _dones = [np.zeros([self.all_num], dtype=np.uint8)]
        if self.network_type == 'gail':
            self.reward_dim_num += 1
            if self.config_nn.GAN_VALUE_TRICK:
                self.value_dim_num += 1
                self.discounts_tmp.append(self.config_nn.GAN_DISCOUNT)
                _dones.append(np.zeros([self.all_num], dtype=np.uint8))
        if self.rnd:
            self.reward_dim_num += 1
            if self.config_nn.RND_VALUE_TRICK:
                self.value_dim_num += 1
                self.discounts_tmp.append(self.config_nn.RND_DISCOUNT)
                _dones.append(np.zeros([self.all_num], dtype=np.uint8))

        self.d_reward_decay = self.config_nn.D_REWARD_DECAY
        self.gail_d_reward_coff = lambda x: self.config_nn.D_REWARD_COFF
        # self.gail_d_reward_coff = lambda x: max(self.config_nn.D_REWARD_COFF - self.config_nn.D_REWARD_COFF * x * self.d_reward_decay, 0)
        self.rnd_reward_coff = self.config_nn.RND_REWARD_COFF
        self.ppo_reward_coff = lambda x: self.config_nn.EXTRINSIC_REWARD_COFF
        #self.ppo_reward_coff = lambda x: min(self.config_nn.EXTRINSIC_REWARD_COFF + self.config_nn.D_REWARD_COFF * x * self.d_reward_decay, 1)

        self.discounts = np.array(self.discounts_tmp, dtype=self.model_dtype).reshape([len(self.discounts_tmp), 1])
        self.tempo_discounts = np.logspace(0, 100, 101, base=self.config_nn.EXTRINSIC_DISCOUNT)
        self.dones = np.stack(_dones)

        self.start_score = 0
        self.is_training: bool = not config.TEST

    def _accumulate_rewards(self, experiences: List[Experience], rewards_step: np.ndarray) -> List[Experience]:
        if len(experiences) == 0:
            return []
        # rewards_sum: [rewards_num_dim, all_num]
        rewards_sum = np.zeros_like(rewards_step[0], dtype=self.model_dtype)
        # next_v: [values_num_dim, all_num]
        next_v = experiences[-1].values
        for t in reversed(range(0, len(experiences) - 1)):
            # gae
            rewards_sum *= (1 - experiences[t].dones)
            rewards_sum = self.discounts * self.landa * rewards_sum + \
                (self.discounts * next_v * (1 - experiences[t].dones) - experiences[t].values + rewards_step[t])
            next_v = experiences[t].values

            experiences[t].values = experiences[t].values + rewards_sum
            experiences[t].advs = rewards_sum[0] * 1.0
        return experiences[:-1]

    def _accumulate_tempo_rewards(self, experiences: List[Experience]) -> List[Experience]:
        if len(experiences) == 0:
            return []
        # rewards_sum: [rewards_num_dim, all_num]
        rewards_sum = np.zeros_like(experiences[0].rewards, dtype=self.model_dtype)
        # next_v: [values_num_dim, all_num]
        next_v = experiences[-1].values
        for t in reversed(range(0, len(experiences) - 1)):  
            # mutli agent check down, if one agent is down, turn his gae to 0
            rewards_sum *= (1 - experiences[t].dones)
            # gae
            tempo_discount = self.tempo_discounts[experiences[t].durations[0]]
            rewards_sum = tempo_discount * self.landa * rewards_sum + \
                (tempo_discount * next_v * (1 - experiences[t].dones) - experiences[t].values + experiences[t].rewards)
            next_v = experiences[t].values
            
            experiences[t].values = experiences[t].values + rewards_sum
            experiences[t].advs = rewards_sum[0] * 1.0
        return experiences[:-1]

    def _update_nav_status(self, info: Dict, dones: np.ndarray):
        if self.config_env.get("env_type") != "robot_nav":
            return
        self.status.update_nav_status(info, dones)

    def _update_reward_status(self, dones: np.ndarray, rewards: np.ndarray):
        self.status.update_reward_status(dones, rewards)

    def _get_predictions(self, states: List[np.ndarray]) -> Tuple[bool, List[np.ndarray]]:
        """TODO change the way of getting bytes data"""
        # the first dim of states is all_num =  agent_num_per_env * batch_env_num
        self.pre_queue.put(self.easy_bytes.encode_forward_states(self.int_process_env_id, states))
        fout = self.conn.brpop(self.action_key, timeout=self.time_out * 10)
        if not fout: # if timeout , fout == None TODO 增加失败次数，一次失败可以继承上次动作
            logging.log(logging.ERROR, " GET ACTION FROM FORWARD MODULE TIMEOUT !")
            raise ValueError
        # list_np_data: (actions, old_logps, values)
        list_np_data: List[np.ndarray] = self.easy_bytes.decode_data(fout[1])
        """
          ppo : values.shape: [1, all_num]
          gail without rnd : values.shape: [2, all_num]
          gail + rnd: values.shape: [3, all_num]
          dim of values depends on self.value_dim_num
        """
        return True, list_np_data
    
    # advantage reshape
    def adv_reshape(self, adv):
        # adv: [reward_dim_num, all_num]
        if self.reward_dim_num == 1:
            return adv[0] * 1.0
        if self.reward_dim_num == 2 and self.network_type == 'gail':
            return self.ppo_reward_coff(self.episode) * adv[0] + self.gail_d_reward_coff(self.episode) * adv[-1]
        if self.reward_dim_num == 2 and self.rnd:
            return self.ppo_reward_coff(self.episode) * adv[0] + self.rnd_reward_coff * adv[1]
        if self.reward_dim_num == 3 and self.rnd and self.network_type == "gail":
            return self.ppo_reward_coff(self.episode) * adv[0] + self.gail_d_reward_coff(self.episode) * adv[-1] \
                + self.rnd_reward_coff * adv[1]

    def _trans_action(self, raw_actions):
        if self.action_dim > 1:
            actions = raw_actions.reshape(self.all_num, self.action_dim)
            actions_reshape = actions.reshape(self.batch_num_per_env, self.agent_num_per_env, self.action_dim)
        else:
            actions = raw_actions  #
            actions_reshape = actions.reshape(self.batch_num_per_env, self.agent_num_per_env)
        return actions, actions_reshape
    
    def _get_durations(self, info: Dict):
        durations = []
        if self.config_env.get('is_temporl') != None:
            for i in range(len(info[0]['real_action'])):
                durations.append(info[0]['real_action'][i].duration)
        return durations
        
    def run_episode(self):
        #
        states = self.env.reset()
        exps = []
        time_step = 0
        # rewards_sum = np.zeros(self.batch_num_per_env, dtype=self.model_dtype)
        # rewards_episode = np.zeros(self.batch_num_per_env, dtype=self.model_dtype)
        # Drewards_sum = np.zeros(self.batch_num_per_env, dtype=self.model_dtype)
        # Drewards_episode = np.zeros(self.batch_num_per_env, dtype=self.model_dtype)
        rewards_step = np.zeros([self.T + 1, self.reward_dim_num, self.all_num], dtype=self.model_dtype)

        while self.exit_flag.value == b'0':
            is_data, data = self._get_predictions(states)
            actions, actions_reshape = self._trans_action(data[0])
            old_logps = data[1]
            values = data[2].reshape(self.value_dim_num, self.all_num)
            """
            # all_agent_num = batch_num * agent_num_per_env
            # values.shape: [values_num_dim, all_agent_num]
            # actions.shape: [all_agent_num, action_dim] if action_dim > 1 else [all_agent_num]
            # old_logps.shape: [all_agent_num]
            # rewards.shape: [all_agent_num]
            # done.shape: [all_agent_num]
            """
            if self.mimic_w:
                self.mimic_w.put(states, actions, self.str_process_env_id)
            next_states, rewards, dones, info = self.env.step(actions_reshape)

            self.dones[0] = dones
            if self.render:
                self.env.render()

            self._update_reward_status(info['all_down'], rewards)
            self._update_nav_status(info, dones)

            rewards_step[time_step][0] = rewards

            # check RND， reshape last state's reward
            if self.rnd:
                # rnd trick, do not consider episode dones
                if time_step > 0:
                    # time_step - 1 : because that rnd_reward calculated by next states
                    rewards_step[time_step-1][1] = data[2]

            # check GAIL , D reward
            if self.network_type == 'gail':
                # follow rnd trick, do not consider episode dones
                rewards_step[time_step][-1] = data[-1]

            if self.is_training:
                exps.append(Experience(states=states,
                                   actions=actions,
                                   durations=self._get_durations(info),
                                   rewards=rewards_step[time_step],
                                   old_logps=old_logps,
                                   dones=deepcopy(self.dones),
                                   values=values,
                                   is_clean=info.get('is_clean')))
            if time_step == self.T:
                # cal the discounted reward
                if self.config_env.get('is_temporl')!=None:
                    updated_exps = self._accumulate_tempo_rewards(exps)
                else:
                    updated_exps = self._accumulate_rewards(exps, rewards_step)
                # states, advs, actions, p(a|s), value
                # Q: why batch?
                # A: beacause there are more than one agents in our environments.
                # Avoid data too large, yield pieces of bexp
                bexp_gen = Experience.batch_data_gene(updated_exps)
                yield bexp_gen, \
                      rewards_step,

                rewards_step[0] = rewards_step[self.T]
                if len(exps) > 0:
                    exps = [exps[-1]]
                time_step = 0

            # Drewards_sum *= (1 - dones)
            states = next_states
            time_step += 1

    def run(self):
        steps = 0
        start_time = time.time()

        # add start_score to tensorboard
        r_dict = {str(self.int_process_env_id*self.batch_num_per_env+i): self.start_score for i in range(self.batch_num_per_env)}
        self.logger_f.add((r_dict, 0), "RewardEpisode")

        for bexp_gen, rewards_step in self.run_episode():
            # put the training data to redis
            steps += self.T
            
            for bexp in bexp_gen:
                self.train_queue.put(bexp)
            end_time = time.time()
            traj_time = end_time - start_time
            self.logger_f.add((traj_time, steps), "TrajectoryTime")
            start_time = end_time

            self.episode += 1
            if self.episode % self.r_logfre == 0:
                self.status.update_reward_logger(steps=steps, int_process_env_id=self.int_process_env_id, config_env=self.config_env)
                if self.config_env.get("env_type") == "robot_nav":
                    self.status.update_action_speed_logger(steps=steps, int_process_env_id=self.int_process_env_id, config_env=self.config_env)
                    # self.status.update_trajectory_steps_logger(steps=steps, int_process_env_id=self.int_process_env_id, config_env=self.config_env)
                    self.status.update_reach_logger(steps=steps, int_process_env_id=self.int_process_env_id, config_env=self.config_env)
                    self.status.update_collision_logger(steps=steps, int_process_env_id=self.int_process_env_id, config_env=self.config_env)
                    if self.config_env['ped_sim']['total'] > 0:
                        self.status.update_ped_relation_velocity_logger(steps=steps, int_process_env_id=self.int_process_env_id, config_env=self.config_env)

                if self.network_type in ['gail', ]:
                    Drewards_step = rewards_step[:, -1, :]
                    #Dre_dict = {str(self.process_env_id * self.batch_num_per_env + i): np.mean(Drewards_episode[i]) for i in range(self.batch_num_per_env)}
                    #self.logger_f.add((Dre_dict, steps), "D_RewardEpisode")
                    for t in range(self.T):
                        Drt_dict = {str(self.int_process_env_id * self.batch_num_per_env + i): np.mean(Drewards_step[t][i]) for i in range(self.batch_num_per_env)}
                        self.logger_f.add((Drt_dict, steps + t - self.T), "GAN[D]_RewardStep")
                if self.rnd:
                    rnd_rewards_step = rewards_step[:, 1, :]
                    for t in range(self.T):
                        intrir_dict = {str(self.int_process_env_id * self.batch_num_per_env + i): np.mean(rnd_rewards_step[t][i]) for i in range(self.batch_num_per_env)}
                        self.logger_f.add((intrir_dict, steps + t - self.T), "Rnd_RewardStep")
            self.train_queue.put(self.status.group_logger(self.config_env))











