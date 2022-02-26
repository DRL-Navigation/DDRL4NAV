#!/bin/bash


source redisc/hosts.sh
#remote_file="/home/drl/${USER}/USTC_lab/sh/redisc"

hosts=(${REDIS_TRAIN_HOST} ${REDIS_PRE_HOST} ${REDIS_MID_HOST} ${REDIS_CON_HOST})
ports=(${REDIS_TRAIN_PORT} ${REDIS_PRE_PORT} ${REDIS_MID_PORT} ${REDIS_CON_PORT})
tags=("train data" "pre data" "middle network module data" "control data")
TASK_NAME=$1



# delete cached redis keys
for ((i=0; i<4; i++));
do
    REDIS_HOST=${hosts[$i]}
    REDIS_PORT=${ports[$i]}
    tag=${tags[$i]}
    echo "Clearing ${tag}"
    str_tmp1=`redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} KEYS "${TASK_NAME}*"`
    if [ "${str_tmp1}" == "" ]
    then
      echo "redis -h ${REDIS_HOST} -p ${REDIS_PORT} is clear"
    else
      redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} KEYS "${TASK_NAME}*" | xargs redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} DEL > /dev/null
    fi

done