import math
import logging
import torch

from USTC_lab.config import game_type


class BaseConfig:
    def __init__(self, parse, config_env: dict):
        if config_env['env_type'] == 'gym':
            self.TASK_TYPE = game_type(config_env['env_name'])
        else:
            self.TASK_TYPE = config_env['env_type']
        self.ENV_NUM = int(config_env['env_num'])
        # self.BATCH_ENV = parse.bn
        # self.ENV_PER_AGENT = parse.env_per_agentnum

        # redis
        self.TRAINER_REDIS_HOST = parse.th
        self.TRAINER_REDIS_PORT = parse.tp
        self.PREDICTOR_REDIS_HOST = parse.ph
        self.PREDICTOR_REDIS_PORT = parse.pp
        self.MIDDLE_REDIS_HOST = parse.mh
        self.MIDDLE_REDIS_PORT = parse.mp
        self.CONTROL_REDIS_HOST = parse.ch
        self.CONTROL_REDIS_PORT = parse.cp

        self.SAVE_MODEL_PATH = parse.model_dir
        self.PREDICTING_MIN_BATCH = math.ceil(self.ENV_NUM / 2)
        self.MACHINE_IP = parse.ip
        if self.MACHINE_IP == "localhost":
            self.MACHINE_IP = "127.0.0.1"
        assert len(self.MACHINE_IP.split(".")) == 4
        self.TASK_NAME = parse.task + "-" + self.MACHINE_IP

    # sync or async, Notion!!: sync supports one process only
    SYNC = False

    # Enable to see the trained agent in action
    PLAY_MODE = False # TODO already out, to delete later
    # Demonstrate
    DEMONSTRATE_MODE = False
    # Load demonstrator path , "redis://{key_name}" "file://{file_path}"
    DEMONSTRATE_LOAD_PATH = "redis://PongDeterministic-v4_v3gail-10iter-1024minbatch-128step-lr0.0002-ent0.01-v12MODEL"
    DEMONSTRATE_SAVE_PATH = "/home/drl/drlnav_frame/mimic/Pong-v4-21-v1-16env/"

    # Mimic
    MIMIC_START = False
    MIMIC_START_LOAD_PATH = "/home/drl/drlnav_frame/mimic/Pong-v4-21-v1-16env/"

    # Load old models. Throws if the model doesn't exist
    LOAD_CHECKPOINT = False
    LOAD_CHECKPOINT_PATH = "/home/ustc/qiuqc/drlnav_master/drlnav_frame/output/robotnav_ped-10obs-5ped-noignore-0.4hz/model/_36000.pt"#"/home/ustc/qiuqc/drlnav_master/drlnav_frame/output/robotnav_test_0.1_hz-2max_acc-15obs-2wacc/model/_20000.pt"#"/home/ustc/qiuqc/drlnav_master/drlnav_frame/output/robotnav_test_0.1_hz-2max_acc-15obs/model/_48000.pt"
    # If 0, the latest checkpoint is loaded
    LOAD_EPISODE = 62000

    # Use RND to expand exploration
    USE_RND = False
    # If the dynamic configuration is on, these are the initial values.
    # num of Thread or Process for Predictors
    PREDICTORS = 1
    # num of Thread or Process for Trainers
    TRAINERS = 1

    # Tmax
    TIME_MAX = 256

    # max broke time
    TIME_OUT = 10

    #########################################################################
    # Number of agents, predictors, trainers and other system settings

    # If the dynamic configuration is on, these are the initial values.

    # QUEUE NUM in ProcessAgents instance
    PREDICTING_QUEUE_NUM = 1

    TRAINING_QUEUE_NUM = 1

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Total number of episodes and annealing frequency
    EPISODES = 400000
    ANNEALING_EPISODE_COUNT = 400000

    #########################################################################
    # Log and save
    # Enable to save models every SAVE_FREQUENCY episodes
    SAVE_MODELS = True
    # Save every SAVE_FREQUENCY episodes
    SAVE_FREQUENCY = 2000

    # Print stats every PRINT_STATS_FREQUENCY episodes
    PRINT_STATS_FREQUENCY = 1
    # The window to average stats
    STAT_ROLLING_MEAN_WINDOW = 1000


    """
       Log info
    """
    INFO_LEVEL = logging.INFO
    LOG_REWARD_FREQUENCY = 1

    LOG_REWARDQSIZE = 50000
    LOG_LOSS_FREQUENCY = 1

    """
        redis key
    """
    TRAINING_DATA_KEY = "TRAIN"

    EXIT_KEY = "EXIT"

    ENV_NUM_KEY = "ENV_NUM"

    TRAINERS_NUM_KEY = "TRAINER_NUM"

    PREDICTORS_NUM_KEY = "PREDICTOR_NUM"

    PREDICTING_STATES_KEY = "FORWARD_STATES"

    PRE_ACTIONS_KEY = "PRE_ACTION_{}"

    MODULE_KEY = "MODEL"

    REWARD_SHOW_KEY = "SHOW_REWARD"

    TRAIN_LOCK_KEY = "LOCK_KEY"

    ENV_NUM_DICT_KEY = "ENV_DICT"

    UPDATE_TAG_KEY = "UPDATE_TAG"

    IMITATION_MODEL_KEY = "MODEL_IMITATION"

    RENDER_KEY = "RENDER_{}"
    RENDER = 1 if PLAY_MODE or DEMONSTRATE_MODE else 0
    #####################################################################################
    # following params will be updated in main.py, do not set them in this file
    #####################################################################################

    ENV_NUM = 0

    BATCH_ENV = 0

    ENV_PER_AGENT = 0
    # Min size of the forward states transfering to GPU
    PREDICTING_MIN_BATCH = math.ceil(ENV_NUM / 2)

    # redis
    TRAINER_REDIS_HOST = None
    TRAINER_REDIS_PORT = None

    PREDICTOR_REDIS_HOST = None
    PREDICTOR_REDIS_PORT = None

    MIDDLE_REDIS_HOST = None
    MIDDLE_REDIS_PORT = None

    CONTROL_REDIS_HOST = None
    CONTROL_REDIS_PORT = None

    SAVE_MODEL_PATH = None

    # TAG = '-{iter}iter-{b}minbatch-{T}step-lr{lr}-ent{ent}-v12'.format_map(
    #     {
    #         'iter': ConfigNN.TRAINING_ITER_TIME,
    #         'b': ConfigNN.TRAINING_MIN_BATCH,
    #         'T': TIME_MAX,
    #         'lr': ConfigNN.LEARNING_RATE,
    #         # 'env': ENV_NUM,
    #         # 'p': PREDICTORS,
    #         # 't': TRAINERS,
    #         # 'be': BATCH_ENV,
    #         'ent': ConfigNN.ENTROPY_LOSS_THETA
    #
    #     }
    # )
    MACHINE_IP = "127.0.0.1"
    # TASK_NAME = BaseConfig.GAME_NAME + TAG
    TASK_NAME = None