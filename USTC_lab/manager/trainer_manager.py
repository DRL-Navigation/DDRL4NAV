from multiprocessing import Value

from USTC_lab.manager import Manager
from USTC_lab.server import BackwardThread
from USTC_lab.data import EasyBytes
from USTC_lab.runner import create_net


class TrainerManager(Manager):
    def __init__(self, *args):
        super(TrainerManager, self).__init__(*args)
        self.configs = args[-1]
        self.net = create_net(self.configs )

        config = self.configs['config']
        if not config.DEMONSTRATE_MODE and not config.PLAY_MODE:
            self.item_num = config.TRAINERS

    def add_tasks(self, net_type: str = "ppo"):
        logger = self.logc.add_logger(net_type + 'backward')
        self.tasks.append(BackwardThread(net=self.net,
                                         trainer_id=len(self.tasks),
                                         logger=logger,
                                         easy_bytes=EasyBytes(),
                                         exit_flag=Value('c', b'0'),
                                         configs=self.configs))
        self.tasks[-1].start()
