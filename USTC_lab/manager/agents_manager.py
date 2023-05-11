import time
import redis

from redis import Redis
from multiprocessing import Value



from USTC_lab.agent import *
from USTC_lab.manager import Manager
from USTC_lab.data import MimicExpFactory
from USTC_lab.data import LoggerControl, EasyBytes
from USTC_lab.env import make_env, make_vecenv


class AgentsManager(Manager):
    def __init__(self, *args):
        super(AgentsManager, self).__init__(*args)
        configs = args[-1]
        config = configs['config']
        config_nn = configs['config_nn']
        config_env = configs['config_env']

        self.item_num = config_env['env_num']
        if config.DEMONSTRATE_MODE:
            # TODO need reconstruct. now only support atari game
            # conn_module = Redis(config.MIDDLE_REDIS_HOST, config.MIDDLE_REDIS_PORT)
            mimic_factory = MimicExpFactory()
            self.mimic_writer = mimic_factory.mimic_writer(config.TASK_TYPE,
                                                      config.TASK_NAME,
                                                      config.DEMONSTRATE_SAVE_PATH,
                                                      config_env["env_num"],
                                                      config_env["int_frame_stack"])

        self.easy_bytes = EasyBytes()
        self.pipe = self.conn.pipeline()

        self.conn_pre = redis.Redis(config.PREDICTOR_REDIS_HOST, config.PREDICTOR_REDIS_PORT)
        self.pipe_pre = self.conn.pipeline()

        self.conn_train = redis.Redis(config.TRAINER_REDIS_HOST, config.TRAINER_REDIS_PORT)
        self.pipe_train = self.conn_train.pipeline()

        self.render_key = config.TASK_NAME + config.RENDER_KEY
        # self.render = config.RENDER
        # self.batch_num = config.BATCH_ENV
        # self._init_render()

        self.pre_queues = [PredictingQueue(i) for i in range(config.PREDICTING_QUEUE_NUM)]
        self.train_queues = [TrainingQueue(i) for i in range(config.TRAINING_QUEUE_NUM)]
        self.pre_tasks, self.train_tasks = [], []
        self.exit_object = object()

        self.config = config
        self.config_nn = config_nn
        self.config_env: dict = config_env

        self._init_queue_task()
        self.start_queue()

    def _init_queue_task(self):
        for qp, qt in zip(self.pre_queues, self.train_queues):
            self.pre_tasks.append(PredictingProcess(q=qp,
                                                    easy_bytes=self.easy_bytes,
                                                    min_batch=self.config.PREDICTING_MIN_BATCH,
                                                    conn=self.conn_pre,
                                                    key=self.config.TASK_NAME + self.config.PREDICTING_STATES_KEY,
                                                    exit_flag=Value('c', b'0'),
                                                    exit_object=self.exit_object))
            self.train_tasks.append(TrainingProcess(q=qt,
                                                    easy_bytes=self.easy_bytes,
                                                    min_batch=self.config_nn.TRAINING_MIN_BATCH,
                                                    conn=self.conn_train,
                                                    key=self.config.TASK_NAME + self.config.TRAINING_DATA_KEY,
                                                    exit_flag=Value('c', b'0'),
                                                    exit_object=self.exit_object))

    def start_queue(self):
        self._start_predicting_queue()
        self._start_training_queue()

    def close_queue(self):
        self._close_predicting_queue()
        self._close_training_queue()

    def _close_predicting_queue(self):
        for q in self.pre_queues:
            q.put(self.exit_object)
        for task in self.pre_tasks:
            task.exit_flag.value = b'1'

    def _close_training_queue(self):
        for q in self.train_queues:
            q.put(self.exit_object)
        for task in self.train_tasks:
            task.exit_flag.value = b'1'

    def _start_predicting_queue(self):
        for task in self.pre_tasks:
            task.start()

    def _start_training_queue(self):
        for task in self.train_tasks:
            task.start()

    def _init_render(self):
        for task in self.tasks:
            self.pipe.set(self.render_key.format(task.str_process_env_id), self.render)
        self.pipe.execute()

    def _check_render(self):
        for task in self.tasks:
            self.pipe.get(self.render_key.format(task.str_process_env_id))
        p = self.pipe.execute()
        for i in range(len(p)):
            if p[i] and int(p[i]) == 1:
                self.tasks[i].render = 1
            else:
                self.tasks[i].render = 0
    
    def _make_vec_env(self):
        envs = [make_env(self.config_env) for i in range(self.config_env.get('batch_num_per_env', 1))]
        # envs = make_env(self.config_env)
        vector_envs = make_vecenv(envs)
        
        return vector_envs
    
    def add_tasks(self, net_type: str = "ppo"):
        process_env_id: str = self.config.MACHINE_IP + "_" + str(self.config_env["node_id"])
        # TODO 直接传进去一个构造好的env
        self.tasks.append(Agents(process_env_id=process_env_id,
                                 logger=self.logc.add_logger('agent'),
                                 easy_bytes=EasyBytes(),
                                 vector_envs=self._make_vec_env(),
                                 pre_queue=self.pre_queues[len(self.tasks) % len(self.pre_queues)],
                                 train_queue=self.train_queues[len(self.tasks) % len(self.train_queues)],
                                 exit_flag=Value('c', b'0'),
                                 mimic_w=self.mimic_writer,
                                 config=self.config,
                                 config_nn=self.config_nn,
                                 config_env=self.config_env,
                                 )
                          )
        self.tasks[-1].start()
        print("add env process {} and start successfully".format(process_env_id), flush=True)

    def pop_all(self):
        self.close_queue()
        super(AgentsManager, self).pop_all()

    def run(self):
        self._check_render()
        return super(AgentsManager, self).run()
