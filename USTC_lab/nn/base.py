# -*- coding: utf-8 -*-
# @Time    : 2021/2/23 4:29 下午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: base.py
import time
import struct
import redis
import torch.nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple
import torch.utils.data as Data
import logging

def mul(x):
    o = 1
    for i in x:
        o *= i
    return o


class Basenn(torch.nn.Module):
    def __init__(self, config, config_nn):
        super(Basenn, self).__init__()
        self.conn = self._connect_redis(config.MIDDLE_REDIS_HOST,
                                        config.MIDDLE_REDIS_PORT)
        self.pipe = self.conn.pipeline()
        self.model_key = config.TASK_NAME + config.MODULE_KEY

        self.device = config.DEVICE

        self.model_dtype = config_nn.MODULE_NUMPY_DTYPE
        self.model_dtype_bytes = config_nn.MODULE_BITS // 8

        self.model_tensor_dtype = config_nn.MODULE_TENSOR_DTYPE

    def _encode_wb(self, wb_np: np.ndarray) -> bytes:
        out = b""
        shape = wb_np.shape
        shape_dim = len(shape)
        out += struct.pack(">I", shape_dim)
        out += struct.pack(">" + "I" * shape_dim, *shape)
        out += wb_np.tobytes()
        return out

    def _decode_wb(self, wb_bytes: bytes) -> Tuple[torch.tensor, int]:
        shape_dim = struct.unpack(">I", wb_bytes[:4])[0]
        shape = struct.unpack(">" + "I" * shape_dim, wb_bytes[4: 4 + shape_dim * 4])
        count = mul(shape)
        wb = np.frombuffer(wb_bytes, dtype=self.model_dtype,
                           offset=4 + shape_dim * 4, count=count).reshape(*shape)

        return torch.tensor(wb, device=self.device), 4 + shape_dim * 4 + count * self.model_dtype_bytes

    def _connect_redis(self, host, port):
        # todo 增加连接成功验证
        return redis.Redis(host=host, port=port)

    def nn2redis(self, pipe, update_key, key=None):
        model_bytes = b""
        for k, v in self.named_parameters():
            model_bytes += self._encode_wb(v.cpu().detach().numpy())
        pipe.set(key if key else self.model_key, model_bytes)
        pipe.incr(update_key)
        pipe.execute()

    def updatenn_by_redis(self, conn, key=None):
        # cost 0.003s one thread
        model_dict = {}
        # 0.005s depends on redis requirements and cpu of redis server

        model_bytes = conn.get(key if key else self.model_key)
        index = 0

        # NOTION: decode costs 0.02s-0.03s in atari.
        for k, v in self.named_parameters():
            k_param, c = self._decode_wb(model_bytes[index:])
            index += c
            model_dict[k] = k_param
        self.load_state_dict(model_dict, strict=False)

    def updatenn_by_file(self, file_path: str):
        self.load_state_dict(torch.load(file_path))
        pass

    # TODO support loading parameters
    def updatenn(self, path: str, conn=None):
        if path.startswith("redis"):
            assert conn is not None
            # redis default host : config.MIDDLE_REDIS_HOST
            self.updatenn_by_redis(conn, path.split("://")[-1])
        # support load local model only
        elif path.startswith("file"):
            self.updatenn_by_file(path.split("://")[-1])

    def init_weight(self):
        for p in self.modules():
            if isinstance(p, torch.nn.Conv2d) or isinstance(p, torch.nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def states_normalization(self, states):
        """
            normalize states
        """
        pass

    def imitation_learning(self,
                           *args,
                           **kwargs):
        if kwargs["imitation_training_type"] == "classification":
            self._imitation_learning_classifier(*args, **kwargs)
        elif kwargs["imitation_training_type"] == "regression":
            self._imitation_learning_regression(*args, **kwargs)

    def _imitation_learning_regression(self, *args, **kwargs):
        pass

    def _imitation_learning_classifier(self,
                                       dataset: Data.Dataset,
                                       pipe: redis.client.Pipeline,
                                       update_key: str,
                                       **kwargs):
        """
            imitation learning , use (st, at) from demonstrator
        """
        loader = Data.DataLoader(dataset, batch_size=kwargs['imitation_training_batch'], shuffle=True)

        optim = torch.optim.Adam(self.parameters(), kwargs['imitation_learning_rate'])
        criterion = torch.nn.CrossEntropyLoss()
        # loop training epochs
        epoch = 0
        while epoch < kwargs["imitation_training_epoch"]:
            epoch += 1
            logging.info("======================================\n")
            logging.info("training epoches : {}".format(epoch))
            logging.info("======================================\n")

            for batch_index, batch in enumerate(loader):
                X, Y = batch[0], batch[1]
                # loss = torch.mean((self(X) - Y) ** 2) / 2
                loss = criterion(self(X.to(self.model_tensor_dtype)), Y.to(self.model_tensor_dtype))
                logging.info("Batch_Index = {}, Loss = {}".format(batch_index, loss.item()))

                optim.zero_grad()
                loss.backward()
                optim.step()
            if epoch % kwargs["imitation_saving_frequency"] == 0:
                self.nn2redis(pipe, update_key, kwargs["imitation_model_key"])


class PreNet(torch.nn.Module):
    def __init__(self):
        super(PreNet, self).__init__()
