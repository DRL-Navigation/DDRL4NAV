import torch, numpy

from USTC_lab.nn import GaussionActor, CategoricalActor

class ConfigNN:
    """
        we need env config to define action dim of critic net.

    """
    def __init__(self, dict_config_env: dict):
        if dict_config_env['discrete_action']:
            self.ACTOR_CLASS = CategoricalActor
            self.ACTION_OUTPUT_DIM = len(dict_config_env['discrete_actions'])
            self.ACTIONS_DIM = 1
        else:
            self.ACTION_OUTPUT_DIM = self.ACTIONS_DIM = dict_config_env['act_dim']
            self.ACTOR_CLASS = GaussionActor
    # ["gail", "ppo"]
    NETWORK_TYPE = "ppo"
    # Use RND to expand exploration
    USE_RND = False

    AC_INPUT_DIM = 512
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Discount factor
    EXTRINSIC_DISCOUNT = 0.99
    # GAE factor
    LANDA = 0.95

    # Learning rate
    LEARNING_RATE = 2e-4
    ACTOR_LEARNING_RATE = 5e-5
    CRITIC_LEARNING_RATE = 1e-3

    V_LOSS_THETA = 1.0

    ENTROPY_LOSS_THETA = 0.05

    PPO_CLIP = 0.2

    # juewu DUEL PPO  2 or 3 in general
    DUEL_PPO_CLIP = 3

    TRAINING_ITER_TIME = 10

    TRAINING_MIN_BATCH = 1024

    # last layer + SOFT_MAX to get classifier, whether to add the softmax grid
    SOFT_MAX_GRID = True

    CLIP_GRID = True
    CLIP_GRID_NUM = 0.5

    SMOOTH_L1_LOSS = False
    # Share image feature net work in actor and critic
    SHARE_CNN_NET = False

    # Grid accumulation steps
    GRAD_ACCUMULATION_STEP = 5

    EXTRINSIC_REWARD_COFF = 0
    """
        MODULE_Precision
    """
    # model.half() 32 -> 16 # https://muzhan.blog.csdn.net/article/details/112472123
    HALF = False
    MODULE_TENSOR_DTYPE = torch.float32 if not HALF else torch.float16
    MODULE_NUMPY_DTYPE = numpy.float32 if not HALF else numpy.float16
    MODULE_BITS = 32 if not HALF else 16

    """
        Imitation Learing parameters
    """
    IMITATION_LEARINING_RATE = 0.0001

    IMITATION_TRAINING_EPOCH = 10000

    IMITATION_TRAINING_BATCH = 1024

    IMITATION_SAVING_FREQUENCY = 100
    # classification or regression
    IMITATION_TRAINING_TYPE = "classification"

    """
        GAN parameters
    """

    # different value network with ppo value
    GAN_VALUE_TRICK = True
    GAN_DISCOUNT = 0.99

    GAN_D_LEARNING_RATE = 5e-5

    GAN_D_EPOCH = 1

    GAN_D_BATCH_SIZE = 128

    WGAN_CLIP_GRAD_NUM = 0.01

    # if D_REWARD_DECAY not 0 : add l1->1  reduce l2->0
    # or 0 : no decay
    D_REWARD_DECAY = 1 / 10000
    D_REWARD_COFF = 1

    # D_reward_function, reference: https://arxiv.org/abs/2104.06687
    # used in class instance, lambda should add more input param "self"
    D_REWARD_FUNCTION_LIST = [
        lambda self, x: x,
        lambda self, x: torch.sqrt(x),
        lambda self, x: torch.exp(x),
    ]
    D_REWARD_FUNCTION = D_REWARD_FUNCTION_LIST[0]


    """
        RND parameters
    """
    # different value network with ppo value
    RND_VALUE_TRICK = False
    RND_DISCOUNT = 0.99

    RND_LEARNING_RATE = 5e-5
    RND_EPOCH = 4
    RND_REWARD_COFF = 1
    # EXTRINSIC_REWARD_COFF = 2

    # GAN_D_LANDA = 0.01

    # the frequency to update model to redis, generally same as ppo training epoch
    MODEL_TO_REDIS_FREQUENCY = GAN_D_EPOCH if NETWORK_TYPE == "gail" else TRAINING_ITER_TIME