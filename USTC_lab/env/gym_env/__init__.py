import gym

from USTC_lab.env.gym_env.wrapper.warputils import *
from USTC_lab.env.gym_env.wrapper.filter_states import *

wrapper_dict = {
    "WarpReward": WarpReward,
    "WarpDone": WarpDone,
    "TimeLimitWrapper": TimeLimitWrapper,
    "NeverStopWrapper": NeverStopWrapper,
    "WarpFrameWrapper": WarpFrameWrapper,
    "FrameStackWrapper": FrameStackWrapper,
    "FilterStatesWrapper": FilterStatesWrapper,
    "WrapAction": WrapAction,
    "ExpandWrapper": ExpandWrapper,
    'DisplayWrapper': DisplayWrapper,
    'InfoExpandWrapper': InfoExpandWrapper,
}


def make_gymenv(cfg):
    env = gym.make(cfg['env_name'])



    for wrapper in cfg['wrapper']:
        env = wrapper_dict[wrapper](env, cfg)
    return env