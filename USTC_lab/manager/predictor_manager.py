from multiprocessing import Value

from USTC_lab.manager import Manager
from USTC_lab.server import ForwardThread
from USTC_lab.data import EasyBytes
from USTC_lab.runner import create_net


class PredictorManager(Manager):
    def __init__(self, *args):
        super(PredictorManager, self).__init__(*args)

        self.configs = args[-1]
        self.net = create_net(self.configs)

        self.item_num = self.configs['config'].PREDICTORS

    def add_tasks(self, net_type: str = ""):
        logger = self.logc.add_logger('forward')
        self.tasks.append(ForwardThread(net=self.net,
                                        predictor_id=len(self.tasks),
                                        logger=logger,
                                        easy_bytes=EasyBytes(),
                                        exit_flag=Value('c', b'0'),
                                        configs=self.configs
                                        ))
        self.tasks[-1].start()
