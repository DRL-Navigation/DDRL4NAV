import time

from torch.utils.tensorboard import SummaryWriter
from typing import *

from USTC_lab.data import *


class LoggerFactory:
    def __init__(self, **kwargs):
        # self.logger_class = []
        self.logger_dict = {}

        self._init_logger(**kwargs)
        # print(self.logger_dict)

    def _init_logger(self, **kwargs):
        for lc in self.logger_class:
            logger: Logger = lc(**kwargs)
            self.logger_dict[str(logger)] = logger

    def add(self, item, tag):
        self.logger_dict[tag].add(item)

    def update_tensorboard(self, logc):
        for logger in self.logger_dict.values():
            logger.update_tensorboard(logc)

    def update_tensor_tags(self, *args):
        for logger in self.logger_dict.values():
            logger.update_tensortags(*args)


class ForwardLogger(LoggerFactory):
    def __init__(self, **kwargs):
        self.logger_class = [ForwardTimeLogger]
        super(ForwardLogger, self).__init__(**kwargs)


class BackwardLogger(LoggerFactory):
    def __init__(self, **kwargs):
        self.logger_class = [TotalLossLogger, ActorLossLogger, VLossLogger, EntLossLogger, BackUpTimeLogger,
                             ]
        super(BackwardLogger, self).__init__(**kwargs)


class PpoBackwardLogger(LoggerFactory):
    """ use this now """
    def __init__(self, **kwargs):
        self.logger_class = [PpoTotalLossLogger, ActorLossLogger, VLossLogger, EntLossLogger, PpoBackUpTimeLogger,
                             RNDLossLogger, RNDBackupTimeLogger,
                             ReducedRewardLogger, ReducedReachRateLogger, ReducedStaticObsCollisionRateLogger, ReducedPedCollisionRateLogger, ReducedOtherRobotCollisionRateLogger,
                             ReducedOpenAreaAngularVelocityEpisodeLogger, ReducedOpenAreaLinearVelocityEpisodeLogger, ReducedStepsEpisodeLogger,
                             ReducedCloseHumanLinearVelocityEpisodeLogger, ReducedCloseHumanAngularVelocityEpisodeLogger,
                             ReducedLinearVelocityEpisodeLogger, ReducedAngularVelocityEpisodeLogger]
        super(PpoBackwardLogger, self).__init__(**kwargs)


class GailBackwardLogger(LoggerFactory):
    """ GAIL LOGGER , PPO + GAIL"""
    def __init__(self, **kwargs):
        self.logger_class = [PpoTotalLossLogger, ActorLossLogger, VLossLogger, EntLossLogger, GAN_D_LossLogger,
                             GAN_D_BackUpTimeLogger, PpoBackUpTimeLogger, RNDLossLogger, RNDBackupTimeLogger]
        super(GailBackwardLogger, self).__init__(**kwargs)


class AgentLogger(LoggerFactory):
    def __init__(self, **kwargs):
        self.logger_class = [RewardEpisodeLogger, TrajectoryTimeLogger, DRewardEpisodeLogger, DRewardStepLogger, RndRewardStepLogger,
                             ReachRateEpisodeLogger, StaticObsCollisionRateEpisode, PedCollisionRateEpisode, OtherRobotCollisionRateEpisode,
                             OpenAreaLinearVelocityEpisodeLogger, OpenAreaAngularVelocityEpisodeLogger,
                             CloseHumanLinearVelocityEpisodeLogger, CloseHumanAngularVelocityEpisodeLogger,
                             LinearVelocityEpisodeLogger, AngularVelocityEpisodeLogger]

        super(AgentLogger, self).__init__(**kwargs)





class LoggerControl(SummaryWriter):
    """
    convergence all loggerfactory
    """
    def __init__(self,
                 config,
                 tfboard_dir: str,
                 ):
        env_num: int = config.ENV_NUM
        task_name: str = config.TASK_NAME
        info_level: int = config.INFO_LEVEL
        # tensorboard writer
        self.file_name = tfboard_dir + task_name + str(int(time.time()))
        print("You can input following in your Terminal to open the tensorboard briefly:")
        print("tensorboard --logdir=" + self.file_name, flush=True)

        super(LoggerControl, self).__init__(self.file_name)
        self.env_num: int = env_num
        self.logger_fs: List[LoggerFactory] = []
        self.task_name: str = task_name
        self.info_level = info_level
        self.params = {
            "task_name" : self.task_name,
            "info_level" : self.info_level,
            "file_name" : self.file_name,
            "log_rewardqsize": config.LOG_REWARDQSIZE,
        }

    def add_logger(self, log_type: str = None) -> LoggerFactory:
        if log_type == 'agent':
            logger_f = AgentLogger(**self.params)
        elif log_type == 'backward':
            logger_f = BackwardLogger(**self.params)
        elif log_type == "ppobackward":
            logger_f = PpoBackwardLogger(**self.params)
        elif log_type == "gailbackward":
            logger_f = GailBackwardLogger(**self.params)
        elif log_type == 'forward':
            logger_f = ForwardLogger(**self.params)
        self.logger_fs.append(logger_f)
        return logger_f

    def pop_logger(self):
        self.logger_fs.pop()

    def update_tensorboard(self):
        for logger_f in self.logger_fs:
            logger_f.update_tensorboard(self)