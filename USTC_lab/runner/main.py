# -*- coding: utf-8 -*-
# @Time    : 2021/2/24 9:29 下午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: main.py
'''
local host main script enterpoint
'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="/home/drl/", help="USTC_lab dir")
parser.add_argument("--type", default="env", help="train| env | predict")
parser.add_argument("--th", default="localhost", help="training redis host")
parser.add_argument("--ph", default="localhost", help="predicting redis host")
parser.add_argument("--mh", default="localhost", help="model params redis host")
parser.add_argument("--ch", default="localhost", help="control redis host")
parser.add_argument("--tp", default=6340, help="training redis port", type=int)
parser.add_argument("--pp", default=6379, help="predicting redis port", type=int)
parser.add_argument("--mp", default=6379, help="model redis port", type=int)
parser.add_argument("--cp", default=6379, help="control redis port", type=int)
# parser.add_argument("--an", default=1, help="nums of all env in one machine", type=int)
# parser.add_argument("--bn", default=1, help="nums of batch env in one process", type=int)
# parser.add_argument("--pn", default=1, help="nums of process in one machine", type=int)
# parser.add_argument("--env_per_agentnum", default=1, help='nums of agent in one env', type=int)
parser.add_argument("--ip", default="localhost", help="machine tag")
parser.add_argument("--tfboard_dir", help="tensorboard directory")
parser.add_argument("--bag_dir", default="" ,help="bag directory")
parser.add_argument("--model_dir", default="", help="model save path")
parser.add_argument("--yaml_f", default="test.yaml")
parser.add_argument("--task", default="BreakoutDeterministic-v4_test", help="task tag")

parse = parser.parse_args()
# 这两行是为了处理导入包的路径
import sys
sys.path.append(parse.path + "/drlnav_frame/")
sys.path.append(parse.path + "/drlnav_frame/USTC_lab/env/drlnav_env")

from redis import Redis

from USTC_lab.manager import *
from USTC_lab.data import LoggerControl
from USTC_lab.runner import ini_config, create_net


if __name__ == "__main__":
    config, config_nn, config_env = ini_config(parse)
    configs = {
        "config": config,
        "config_nn": config_nn,
        "config_env": config_env,
    }
    # sync only support one process
    if config.SYNC:
        assert config_env['env_num'] == 1
    conn = Redis(config.CONTROL_REDIS_HOST, config.CONTROL_REDIS_PORT)
    logc = LoggerControl(config=config,
                         tfboard_dir=parse.tfboard_dir + parse.type)

    this_manager: Manager = ManagerFactory(parse.type)(conn, logc, configs)
    this_manager.run()
