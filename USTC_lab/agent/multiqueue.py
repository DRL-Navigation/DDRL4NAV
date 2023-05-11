import numpy as np
from redis import Redis
#
from multiprocessing import Queue as MultiQueue
from multiprocessing import Value
from multiprocess.context import Process
from threading import Thread
from typing import List, Union, Tuple, Dict


from USTC_lab.data import Experience, EasyBytes


class AgentQueue:
    def __init__(self, qid):
        self.qid = qid
        self.q = MultiQueue()

    def put(self, a):
        return self.q.put(a)

    def get(self):
        return self.q.get()


class PredictingQueue(AgentQueue):
    def __init__(self, qid):
        super(PredictingQueue, self).__init__(qid)


class TrainingQueue(AgentQueue):
    def __init__(self, qid):
        super(TrainingQueue, self).__init__(qid)


class QueueProcess(Thread):
    def __init__(self, q: AgentQueue,
                 easy_bytes: EasyBytes,
                 min_batch: int,
                 conn: Redis,
                 key: str,
                 exit_flag: Value,
                 exit_object: object):
        super(QueueProcess, self).__init__()
        self.q = q
        self.easy_bytes = easy_bytes
        self.min_batch = min_batch
        self.conn = conn
        self.key = key

        self.exit_flag = exit_flag
        self.exit_object = exit_object

    def batch(self, t: Union[List[bytes], List[Experience]]) -> bytes:
        pass

    def run(self):
        pass


class TrainingProcess(QueueProcess):
    def __init__(self, **kwargs):
        super(TrainingProcess, self).__init__(**kwargs)

    # TODO try to use memoryview to reduce memory copy
    def batch(self, t: List[Experience]) -> bytes:
        b = b""
        for d in t:
            encoded_d = Experience.encode_data(d)
            b += encoded_d
        return b

    def batch_logger(self, p: List[Dict]) -> Dict:
        if len(p) == 0: return {}
        o = {}
        for key in p[0]:
            t = []
            for j in p:
                t.append(j[key])
            o[key] = float(np.mean(t))
        return o

    def run(self):
        t: List[Experience] = []
        p: List[Dict] = []
        cur_size = 0
        while self.exit_flag.value == b'0':
            c = self.q.get()
            if c is self.exit_object:
                self.q.put(c)
                break
            if isinstance(c, Experience):
                t.append(c)
                cur_size += len(c)
            elif isinstance(c, dict):
                p.append(c)
            else:
                print("something unknown object flow into training pipleine. the type is", type(c))
                raise ValueError
            if cur_size >= 128:
                bytes_data = self.easy_bytes.encode_backward_data(Experience.batch_data(t).get_xrapv(), self.batch_logger(p))
                self.conn.lpush(self.key, bytes_data)
                t.clear()
                p.clear()
                cur_size = 0



class PredictingProcess(QueueProcess):
    def __init__(self, **kwargs):
        super(PredictingProcess, self).__init__(**kwargs)

    def batch(self, t: List[bytes]) -> bytes:
        b = b""
        for i in t:
            b += i
        return b

    def run(self):
        t = []
        while self.exit_flag.value == b'0':
            c = self.q.get()
            if c is self.exit_object:
                self.q.put(c)
                break
            t.append(c)
            if len(t) >= self.min_batch:
                data = self.batch(t)
                self.conn.lpush(self.key, data)
                t.clear()
