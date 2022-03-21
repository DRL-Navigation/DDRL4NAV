#!/bin/bash
####
# @Article{chen2020distributed,
  #title = {Distributed Non-Communicating Multi-Robot Collision Avoidance via Map-Based Deep Reinforcement Learning},
  #author = {Chen, Guangda and Yao, Shunyi and Ma, Jun and Pan, Lifan and Chen, Yu'an and Xu, Pei and Ji, Jianmin and Chen, Xiaoping},
  #journal = {Sensors},
  #volume = {20},
  #number = {17},
  #pages = {4836},
  #year = {2020},
  #publisher = {Multidisciplinary Digital Publishing Institute},
  #doi = {10.3390/s20174836},
  #url = {https://www.mdpi.com/1424-8220/20/17/4836}
  #};
####


# your name
echo "current user: ${USER}"
# task name
# Notion: used in dir name
VERSION=1
# Notion: if you choose to train the atari game, you should set the game name in USTC_lab/config/ATARI_LIST.txt like "Pong-v4"
#TASK_NAME="Pong-v4"
# Notion: if you choose to train robot navigtion, you should set "robotnav"
TASK_NAME="robotnav_${VERSION}"

# mode: "TRAIN" train model or "PLAY" play model
MODE="TRAIN"

# redis file
REDIS_FILE="redisc/hosts.sh"

# machine file
MACHINE_FILE="machines/machines_sensor_map.sh"

# output dir
OUTPUT_DIR="DDRL4NAV/output/"

# python venv file
VENV="/home/drl/drl_venv38/bin/activate"
if [ ! -d ${VENV} ];
then
    VENV="conda_bash.sh"
fi
