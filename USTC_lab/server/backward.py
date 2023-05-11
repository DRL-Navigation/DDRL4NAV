# -*- coding: utf-8 -*-
# @Time    : 2021/2/24 3:52 下午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: backward.py
"""
network backup
"""
import time
import torch
import redis
import numpy as np

from threading import Thread
from multiprocessing import Value
from multiprocess.context import Process
from collections import defaultdict
from multiprocessing import Queue as MultiQueue
from multiprocessing import Value
from typing import *

from USTC_lab.data import Experience
from USTC_lab.data import LoggerFactory, EasyBytes
from USTC_lab.nn import Basenn
from USTC_lab.data import MimicExpFactory


def avg(x):
    return sum(x) / len(x)


def batch_logger( p: List[Dict]) -> Dict:
    if len(p) == 0: return {}
    o = {}
    for key in p[0]:
        t = []
        for j in p:
            t.append(j[key])
        o[key] = float(np.mean(t))
    return o


class BackwardQueue:
    def __init__(self):
        super(BackwardQueue, self).__init__()

        self.q = MultiQueue()

    def get(self, batch_size, *args) -> Tuple[Experience, Dict]:
        cur_size = 0
        list_exp: List[Experience] = []
        list_dict: List[dict] = []
        while cur_size < batch_size:
            data, dict_logger = self.q.get(*args)
            assert isinstance(data, Experience)
            assert isinstance(dict_logger, dict)
            cur_size += len(data)
            list_exp.append(data)
            if len(dict_logger):
                list_dict.append(dict_logger)
        exp = Experience.batch_data(list_exp)
        batched_dict_logger = batch_logger(list_dict)
        return exp, batched_dict_logger

    def put(self, data: Tuple[Experience, Dict], *args) -> None:
        self.q.put(data, *args)


class BackwardThread(Thread):
    def __init__(self,
                 net: Basenn,
                 training_data_queue: BackwardQueue,
                 trainer_id: int,
                 logger: LoggerFactory,
                 easy_bytes: EasyBytes,
                 exit_flag: Value,
                 configs):
        super(BackwardThread, self).__init__()
        # self.setDaemon(True)
        config, config_nn = configs['config'], configs['config_nn']

        self.net = net
        self.training_data_queue = training_data_queue
        self.easy_bytes = easy_bytes
        self.conn_train = redis.Redis(config.TRAINER_REDIS_HOST,
                                      config.TRAINER_REDIS_PORT)
        self.pipe_train = self.conn_train.pipeline()
        self.conn_middle = redis.Redis(config.MIDDLE_REDIS_HOST,
                                       config.MIDDLE_REDIS_PORT)
        self.pipe_middle = self.conn_middle.pipeline()

        self.data_key = config.TASK_NAME + config.TRAINING_DATA_KEY

        self.logger_f = logger
        self.logger_f.update_tensor_tags('train/', trainer_id)
        self.tensortype = config_nn.MODULE_TENSOR_DTYPE
        self.nptype = config_nn.MODULE_NUMPY_DTYPE
        self.device = config_nn.DEVICE
        self.min_batch_size = config_nn.TRAINING_MIN_BATCH

        self.conn_middle.set(config.TASK_NAME + config.TRAIN_LOCK_KEY, 0)
        self.episode = 0
        self.data_len = 0
        self.exit_flag: Value = exit_flag

        self.update_tag = config.TASK_NAME + config.UPDATE_TAG_KEY
        self.train_lock_key = config.TASK_NAME + config.TRAIN_LOCK_KEY

        self.log_loss_freq = config.LOG_LOSS_FREQUENCY
        self.timeout = config.TIME_OUT

        self.save_model: bool = config.SAVE_MODELS
        self.save_freq: int = config.SAVE_FREQUENCY
        self.save_model_path: str = config.SAVE_MODEL_PATH

        self.model2redis_freq = config_nn.MODEL_TO_REDIS_FREQUENCY

        self.mimic_start = config.MIMIC_START
        if config.MIMIC_START:
            self.dataset = MimicExpFactory().mimic_reader(config.TASK_TYPE,
                                                          config.MIMIC_START_LOAD_PATH,
                                                          config_nn.IMITATION_TRAINING_TYPE)
            self.imitation_learning_dict = {
                "imitation_learning_rate": config_nn.IMITATION_LEARINING_RATE,
                "imitation_training_batch": config_nn.IMITATION_TRAINING_BATCH,
                "imitation_training_epoch": config_nn.IMITATION_TRAINING_EPOCH,
                "imitation_saving_frequency": config_nn.IMITATION_SAVING_FREQUENCY,
                "imitation_model_key": config.TASK_NAME + config.IMITATION_MODEL_KEY,
                "imitation_training_type": config_nn.IMITATION_TRAINING_TYPE,
            }

        self.load_checkpoint_path = None
        self.load_checkpoint_start = 0
        if config.LOAD_CHECKPOINT:
            self.load_checkpoint_path = config.LOAD_CHECKPOINT_PATH
            self.load_checkpoint_start += config.LOAD_EPISODE

        self.sync = config.SYNC
        self.test = config.TEST


class BackwardGetDataThread(BackwardThread):
    def __init__(self, *args, **kwargs):
        super(BackwardGetDataThread, self).__init__(*args, **kwargs)

    def get_train_data(self):
        batch_bytes = self.conn_train.brpop(self.data_key, timeout=self.timeout * 3)[1]
        self.conn_middle.set(self.train_lock_key, 1)
        list_np_states, list_np_other4, dict_logger = self.easy_bytes.decode_backward_data(batch_bytes)
        exp_data = Experience(list_np_states, *list_np_other4)

        self.training_data_queue.put( (exp_data, dict_logger) )

    def run(self) -> None:
        while self.exit_flag.value == b'0':
            # get training data from message queue and put them in ProcessQueue.
            self.get_train_data()


class BackwardTrainThread(BackwardThread):
    def __init__(self, *args, **kwargs):
        super(BackwardTrainThread, self).__init__(*args, **kwargs)

    def update_envstats_logger(self, dict_logger):
        for k, v in dict_logger.items():
            self.logger_f.add(({"mean": v}, self.data_len), k)

    # TODO support mutil GPU CARD
    def run(self):
        # while self.exit_mimic.value == b'0':
        if self.mimic_start:
            self.net.imitation_learning(self.dataset,
                                        self.pipe_middle,
                                        self.update_tag,
                                        **self.imitation_learning_dict,)
        if self.load_checkpoint_path:
            self.net.load_state_dict(torch.load(self.load_checkpoint_path))

        loss_dict = defaultdict(list)
        self.net.nn2redis(self.pipe_middle, self.update_tag)
        while self.exit_flag.value == b'0':
            s = time.time()
            train_data, dict_logger = self.training_data_queue.get(self.min_batch_size)
            train_data.to_tensor(dtype=self.tensortype, device=self.device)
            self.data_len += len(train_data)
            print("get training data ", time.time() - s, flush=True)
            self.update_envstats_logger(dict_logger=dict_logger)

            if self.test: continue
            for loss_items, update_time, last in self.net.learn(train_data):
                update_time += self.load_checkpoint_start
                ss = time.time()
                for k, v in loss_items.items():
                    loss_dict[k].append(v)

                # update model to redis
                if last and update_time % self.model2redis_freq == 0:
                    self.net.nn2redis(self.pipe_middle, self.update_tag)

#                self.logger_f.add((once_time, update_time), "BackUpTime")
                # save loss log
                # print("backward", update_time, loss_dict)
                if update_time % self.log_loss_freq == 0:
                    for loss_key, loss_value in loss_dict.items():
                        if len(loss_value):
                            self.logger_f.add((avg(loss_value), update_time), loss_key)
                            loss_dict[loss_key].clear()
                # save model to hard disk
                if last and self.save_model and update_time % self.save_freq == 0:
                    torch.save(self.net.state_dict(), self.save_model_path + "_" + str(update_time) + ".pt")
                # print("save time: ", time.time() - ss, flush=True)

            self.pipe_middle.set(self.train_lock_key, 0)
            self.pipe_middle.execute()

            print("once backward COSTS: ", time.time() - s, flush=True)

        print("backward exit !", flush=True)
