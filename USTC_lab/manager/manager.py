import time
from redis import Redis
from typing import List

from USTC_lab.data import LoggerControl


# def manager(managers: List["Manager"]) -> None:
#     flag = 0
#     while not flag:
#         for m in managers:
#             flag = m.run()
#
#         time.sleep(2)
#     for m in managers:
#         m.pop_all()


class Manager():
    def __init__(self,
                 conn: Redis,
                 logc: LoggerControl,
                 configs=None,
                 ):
        config = configs['config']
        config_nn = configs['config_nn']
        self.conn = conn
        self.tasks = []
        self.exit_key = config.TASK_NAME + config.EXIT_KEY
        self.exit_flag = 0
        # self.item_num_key = item_num_key
        # self.item_num = item_num
        self.item_num = 0
        self.cur_num = 0
        self.logc = logc
        self.mimic_writer = None
        self.net_type = config_nn.NETWORK_TYPE

    @property
    def exit_flag(self):
        return int(self.conn.get(self.exit_key))

    @exit_flag.setter
    def exit_flag(self, _exit_flag):
        self.conn.set(self.exit_key, _exit_flag)

    # @property
    # def item_num(self):
    #     return int(self.conn.get(self.item_num_key))

    """
        the num of [envs | trainers | predictors]
    """
    # @item_num.setter
    # def item_num(self, item_num):
    #     self.conn.set(self.item_num_key, item_num)

    def add_tasks(self, net_type: str = "ppo"):
        pass

    def pop_tasks(self):
        self.tasks[-1].exit_flag.value = b'1'
        self.tasks[-1].join()
        if self.logc:
            self.logc.pop_logger()
        self.tasks.pop()

    def pop_all(self):
        print("game over ... pop all ", flush=True)
        while len(self.tasks):
            self.pop_tasks()

    def run(self):
        while True:
            if self.logc:
                self.logc.update_tensorboard()
            if self.mimic_writer:
                self.mimic_writer.write()
            while self.item_num > self.cur_num:
                # net_type is to choose proper logger_f in LoggerControl instance
                self.add_tasks(self.net_type)
                self.cur_num += 1
            while self.item_num < self.cur_num:
                self.pop_tasks()
                self.cur_num -= 1
            if self.exit_flag:
                break
            time.sleep(2)