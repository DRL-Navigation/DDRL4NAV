import logging
from multiprocessing import Queue
# from queue import Queue

"""
if you want to add a tensorboard logger 
just write one logger Like LossLogger, and then add the class to LoggerFactory
    
such as:
    class LossLogger(Logger):
        self.tensor_tag = 'LossLogger'
        ...
        
    class BackwardLogger(LoggerFactory):
        def __init__(self, **kwargs):
            super(BackwardLogger, self).__init__(**kwargs)
            self.logger_class = [LossLogger, BackUpTimeLogger]
"""


class Logger:
    def __init__(self, **kwargs):
        # super(Logger, self).__init__(kwargs['file_name'])

        self.task_name = kwargs['task_name']
        self.file_name = kwargs['file_name']
        logger = logging.getLogger(kwargs['file_name'])
        logger.setLevel(kwargs['info_level'])
        log_rewardqsize = kwargs['log_rewardqsize']
        self.q = Queue(log_rewardqsize)
        self.tensor_tag = self.task_name + "{}{}_" + str(self)  # env0_RewardEpisode | trainer0_BackUpTime
        # function
        self.f = None

    def add(self, r):
        self.q.put(r)

    def get_all(self):
        size = self.q.qsize()
        for i in range(size):
            yield self.q.get()

    def update_tensorboard(self, logc):
        gen_r = self.get_all()
        for rs in gen_r:
            self.f(logc)(self.tensor_tag, rs[0], rs[1])

    def update_tensortags(self, tag_type: str, env_id: int):
        self.tensor_tag = self.tensor_tag.format(tag_type, env_id)

    def __str__(self):
        raise NotImplemented


class HistogramLogger(Logger):
    def __init__(self, **kwargs):
        super(HistogramLogger, self).__init__(**kwargs)
        self.f = lambda x: x.add_histogram


class ScalarLogger(Logger):
    def __init__(self, **kwargs):
        super(ScalarLogger, self).__init__(**kwargs)
        self.f = lambda x: x.add_scalar


class ScalarsLogger(Logger):
    def __init__(self, **kwargs):
        super(ScalarsLogger, self).__init__(**kwargs)
        self.f = lambda x: x.add_scalars

        self.tfboard_type = 'scalars'
        self.tensor_tag = self.task_name + "_" + str(self) + "/{}{}"


class ReducedScalarLogger(ScalarLogger):
    def __init__(self, **kwargs):
        super(ReducedScalarLogger, self).__init__(**kwargs)
        self.f = lambda x: x.add_scalars

        self.tfboard_type = 'scalars'
        self.tensor_tag = self.task_name + "ReducedLogger/" + str(self)






