#!/bin/bash


source config.sh
source machines/all.sh



if [ "$1" == "remote" ];
then
    file="/home/drl/${USER}/"
else
    file="`pwd`/../../"
fi
pid_dir="${file}${OUTPUT_DIR}/${TASK_NAME}/pid/"


if [ "$1" == "remote" ];
then
    for ENV_MACHINE in ${ENV_MACHINES};
    do
        host="drl@${ENV_MACHINE}"
        ssh ${host} "kill -9 `head -n 1 "${pid_dir}/env.pid"`"
    done

    for PRE_MACHINE in ${PRE_MACHINES};
    do
        host="drl@${PRE_MACHINE}"
        ssh ${host}  "kill -9 `head -n 1 "${pid_dir}/predict.pid"`"
    done

    for TRAIN_MACHINE in ${TRAIN_MACHINES};
    do
        host="drl@${TRAIN_MACHINE}"
        ssh ${host}  "kill -9 `head -n 1 "${pid_dir}/train.pid"`"
    done
else
    kill -9 `head -n 1 "${pid_dir}/env.pid"`
    kill -9 `head -n 1 "${pid_dir}/predict.pid"`
    kill -9 `head -n 1 "${pid_dir}/train.pid"`
fi

