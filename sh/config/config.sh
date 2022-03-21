#!/bin/bash

# your name
echo "current user: ${USER}"
# task name
# Notion: used in dir name
VERSION=1
# Notion: if you choose to train the atari game, you should set the game name in USTC_lab/config/ATARI_LIST.txt like "Pong-v4"
TASK_NAME="Pong-v4"
# Notion: if you choose to train robot navigtion, you should set "robotnav"
#TASK_NAME="robotnav_${VERSION}"

# mode: "TRAIN" train model or "PLAY" play model
MODE="TRAIN"

# redis file
REDIS_FILE="redisc/hosts.sh"

# machine file
MACHINE_FILE="machines/machines_atari.sh"

# output dir
OUTPUT_DIR="DDRL4NAV/output/"

# python venv file
VENV="/home/drl/drl_venv38/bin/activate"
if [ ! -d ${VENV} ];
then
    VENV="conda_bash.sh"
fi
