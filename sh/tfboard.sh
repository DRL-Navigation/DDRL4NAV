#!/bin/bash


source config.sh
source machines/all.sh
if [ "$1" == "remote" ];
then
    file="/home/drl/${USER}/"
else
    file="`pwd`/../../"
fi
tfboard_dir="${file}${OUTPUT_DIR}/${TASK_NAME}/tfboard/"
echo $tfboard_dir
if [ "$1" == "remote" ];
then
    for ENV_MACHINE in ${ENV_MACHINES};
    do
        host="drl@${ENV_MACHINE}"
        tar_f="${tfboard_dir}${ENV_MACHINE}_env.tar.gz"
        ssh ${host}  "tar -czf ${tar_f} ${tfboard_dir}/env*"
        scp ${host}:${tar_f} ${tar_f}
    done

    for PRE_MACHINE in ${PRE_MACHINES};
    do
        host="drl@${PRE_MACHINE}"
        tar_f="${tfboard_dir}${PRE_MACHINE}_pre.tar.gz"
        ssh ${host}  "tar -czf ${tar_f} ${tfboard_dir}/pre*"
        scp ${host}:${tar_f} ${tar_f}
    done

    for TRAIN_MACHINE in ${TRAIN_MACHINES};
    do
        host="drl@${TRAIN_MACHINE}"
        tar_f="${tfboard_dir}${TRAIN_MACHINE}_train.tar.gz"
        ssh ${host}  "tar -czf ${tar_f} ${tfboard_dir}/train*"
        scp ${host}:${tar_f} ${tar_f}
    done
    cd ${tfboard_dir}
    for tar_f in `ls *.tar.gz`
    do
        tar -xzvf ${tar_f}
    done

fi


echo "tensorboard --logdir=${tfboard_dir}"  #--host=0.0.0.0
tensorboard --logdir=${tfboard_dir}