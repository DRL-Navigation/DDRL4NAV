# @Author  : qiuqc@mail.ustc.edu.cn
import yaml
import copy
import gym

from USTC_lab.config import *
from USTC_lab.nn import *


def read_yaml(parse) -> dict:
    file = parse.yaml_f
    file = open(file, 'r', encoding="utf-8")
    # 读取文件中的所有数据
    file_data = file.read()
    file.close()
    # 指定Loader
    dict_original = yaml.load(file_data, Loader=yaml.FullLoader)
    update_cfg_env(dict_original)
    update_bag_path(parse, dict_original)
    check_nav_robot_num(dict_original)

    return dict_original


def check_nav_robot_num(cfg):
    if cfg['env_type'] == 'robot_nav':
        assert cfg['agent_num_per_env'] == cfg['robot']['total']
        assert cfg['cfg_type'] in ['yaml', 'bag']


def update_cfg_env(cfg):
    # used for config_nn
    if cfg['env_type'] == 'gym':
        env = gym.make(cfg['env_name'])

        if type(env.action_space) == gym.spaces.Discrete:
            cfg['discrete_action'] = True
            cfg['discrete_actions'] = list(range(env.action_space.n))
        elif type(env.action_space) == gym.spaces.Box:
            cfg['discrete_action'] = False
            # TODO support more than one dim
            cfg['act_dim'] = env.action_space.shape[0]
        cfg['input_dim'] = env.observation_space.shape[0] * cfg['int_frame_stack']


def update_bag_path(parse, cfg):
    cfg['bag_record_output_name'] = parse.bag_dir + cfg.get('bag_record_output_name', "test.bag")


def ini_config(parse):
    dict_config_env = read_yaml(parse)
    config_nn = ConfigNN(dict_config_env)
    config = BaseConfig(parse, dict_config_env)
    print("TASK NAME", config.TASK_NAME, flush=True)
    config.TEST = dict_config_env.get('test', False)
    return config, config_nn, dict_config_env


def create_net(configs) -> Basenn:
    config, config_nn, config_env = configs['config'], configs['config_nn'], configs['config_env']
    if config.TASK_TYPE in ["mujoco", "classical"]:
        if config_nn.SHARE_CNN_NET:

            actor, critic = config_nn.ACTOR_CLASS(action_output_dim=config_nn.ACTION_OUTPUT_DIM,
                                                  device=config_nn.DEVICE,
                                                  soft_max_grid=config_nn.SOFT_MAX_GRID,
                                                  last_input_dim=config_nn.AC_INPUT_DIM,
                                                  nn_dtype=config_nn.MODULE_TENSOR_DTYPE), \
                            Critic(device=config_nn.DEVICE,
                                   last_input_dim=config_nn.AC_INPUT_DIM)
            # prenet = mlp(config.MLP_LIST)
            prenet = MLPPreNet(config_env.get('input_dim', 4), config_nn.AC_INPUT_DIM)
        else:
            pre_actor = MLPPreNet(config_env.get('input_dim', 4), config_nn.AC_INPUT_DIM)
            pre_critic = MLPPreNet(config_env.get('input_dim', 4), config_nn.AC_INPUT_DIM)

            actor, critic = config_nn.ACTOR_CLASS(action_output_dim=config_nn.ACTION_OUTPUT_DIM,
                                                 device=config_nn.DEVICE,
                                                 soft_max_grid=config_nn.SOFT_MAX_GRID,
                                                 last_input_dim=config_nn.AC_INPUT_DIM,
                                                 nn_dtype=config_nn.MODULE_TENSOR_DTYPE,
                                                  pre=pre_actor), \
                           Critic(device=config_nn.DEVICE,
                                  last_input_dim=config_nn.AC_INPUT_DIM,
                                  pre=pre_critic)
            prenet = None
    elif config.TASK_TYPE in ["robot_nav", "gazebo_env", "real_env"]:
        if config_nn.SHARE_CNN_NET:
            actor, critic = config_nn.ACTOR_CLASS(action_output_dim=config_nn.ACTION_OUTPUT_DIM,
                                                  device=config_nn.DEVICE,
                                                  soft_max_grid=config_nn.SOFT_MAX_GRID,
                                                  last_input_dim=config_nn.AC_INPUT_DIM,
                                                  nn_dtype=config_nn.MODULE_TENSOR_DTYPE), \
                            Critic(device=config_nn.DEVICE,
                                   last_input_dim=config_nn.AC_INPUT_DIM)

            if config_env['ped_sim']['total'] > 0:
                """running in a scenario with several pedestrians. use ped map
                see our early paper: Crowd-Aware Robot Navigation for Pedestrians with Multiple Collision Avoidance Strategies via Map-based Deep Reinforcement Learning
                https://arxiv.org/abs/2109.02541
                """
                prenet = NavPedPreNet(image_channel=config_env["image_batch"]+3, last_output_dim=config_nn.AC_INPUT_DIM)
            else:
                prenet = NavPreNet(image_channel=config_env["image_batch"], last_output_dim=config_nn.AC_INPUT_DIM)
        else:
            # pre_actor = NavPedPreNet(image_channel=config_env["image_batch"]+3, last_output_dim=config_nn.AC_INPUT_DIM)
            # pre_critic = NavPedPreNet(image_channel=config_env["image_batch"]+3, last_output_dim=config_nn.AC_INPUT_DIM)
            # pre_actor = NavPreNet(image_channel=config_env["image_batch"], last_output_dim=config_nn.AC_INPUT_DIM)
            # pre_critic = NavPreNet(image_channel=config_env["image_batch"], last_output_dim=config_nn.AC_INPUT_DIM)
            pre_actor = NavPreNet1D(image_channel=3, last_output_dim=config_nn.AC_INPUT_DIM)
            pre_critic = NavPreNet1D(image_channel= 3, last_output_dim=config_nn.AC_INPUT_DIM)
            actor, critic = config_nn.ACTOR_CLASS(action_output_dim=config_nn.ACTION_OUTPUT_DIM,
                                                  device=config_nn.DEVICE,
                                                  soft_max_grid=config_nn.SOFT_MAX_GRID,
                                                  last_input_dim=config_nn.AC_INPUT_DIM,
                                                  nn_dtype=config_nn.MODULE_TENSOR_DTYPE,
                                                  pre=pre_actor), \
                            Critic(device=config_nn.DEVICE,
                                   last_input_dim=config_nn.AC_INPUT_DIM,
                                   pre=pre_critic)
            prenet = None
    elif config.TASK_TYPE == 'atari':
        if not config_nn.SHARE_CNN_NET:
            pre_actor = AtariPreNet(config_env['int_frame_stack'], last_output_dim=config_nn.AC_INPUT_DIM, device=config_nn.DEVICE)
            pre_critic = AtariPreNet(config_env['int_frame_stack'], last_output_dim=config_nn.AC_INPUT_DIM, device=config_nn.DEVICE)
            actor, critic = config_nn.ACTOR_CLASS(action_output_dim=config_nn.ACTION_OUTPUT_DIM,
                                                  device=config_nn.DEVICE,
                                                  soft_max_grid=config_nn.SOFT_MAX_GRID,
                                                  last_input_dim=config_nn.AC_INPUT_DIM,
                                                  pre=pre_actor,
                                                  nn_dtype=config_nn.MODULE_TENSOR_DTYPE), \
                            Critic(device=config_nn.DEVICE,
                                   last_input_dim=config_nn.AC_INPUT_DIM,
                                   pre=pre_critic)
            prenet = None
        else:
            actor, critic, prenet = config_nn.ACTOR_CLASS(action_output_dim=config_nn.ACTION_OUTPUT_DIM,
                                                          device=config_nn.DEVICE,
                                                          last_input_dim=config_nn.AC_INPUT_DIM,
                                                          soft_max_grid=config_nn.SOFT_MAX_GRID,
                                                          nn_dtype=config_nn.MODULE_TENSOR_DTYPE), \
                                    Critic(device=config_nn.DEVICE), \
                                    AtariPreNet(config_env['int_frame_stack'], last_output_dim=config_nn.AC_INPUT_DIM,
                                                device=config_nn.DEVICE)
    else:
        pass

    rnd_net = None
    if config.USE_RND:
        rnd_critic = copy.deepcopy(critic) if config_nn.RND_VALUE_TRICK else None
        # RND input frame is 1 based on RND paper.
        rnd_net = RND(pre_f=AtariPreNet(1, last_output_dim=config_nn.AC_INPUT_DIM, device=config_nn.DEVICE),
                      pre_p=AtariPreNet(1, last_output_dim=config_nn.AC_INPUT_DIM, device=config_nn.DEVICE),
                      rnd_critic=rnd_critic,
                      config=config,
                      config_nn=config_nn)
    net = None
    # if we are in demonstrate mode, do not need to open train process.
    if config_nn.NETWORK_TYPE == "ppo":
        net = PPO(actor, critic, prenet, rnd_net, config, config_nn).to(config_nn.DEVICE)
    if config_nn.NETWORK_TYPE == "gail":
        gail_critic = copy.deepcopy(critic)
        ppo_net = PPO(actor, critic, prenet, rnd_net, config, config_nn).to(config_nn.DEVICE)
        D_prenet = copy.deepcopy(prenet) if prenet else None
        D_net = Discriminator(pre=D_prenet,
                              config=config,
                              config_nn=config_nn).to(config_nn.DEVICE)
        net = GAIL(generator=ppo_net, discriminator=D_net, gail_critic=gail_critic).to(config_nn.DEVICE)
    assert net is not None
    return net