#!/bin/bash



source config.sh

source ${MACHINE_FILE}
source ${REDIS_FILE}
if [ "$1" == "remote" ];
then
    root_dir="/home/drl/${USER}/"
else
    root_dir="`pwd`/../../"
fi
run_file="${root_dir}/drlnav_frame/USTC_lab/runner/main.py"
pid_dir="${root_dir}${OUTPUT_DIR}${TASK_NAME}/pid/"
log_dir="${root_dir}${OUTPUT_DIR}${TASK_NAME}/log/"
model_dir="${root_dir}${OUTPUT_DIR}${TASK_NAME}/model/"
tfboard_dir="${root_dir}${OUTPUT_DIR}${TASK_NAME}/tfboard/"
bag_dir="${root_dir}${OUTPUT_DIR}${TASK_NAME}/bag/"

if [ ! -d ${pid_dir} ];
then
    mkdir -p ${pid_dir}

fi

if [ ! -d ${log_dir} ];
then
    mkdir -p ${log_dir}

fi

if [ ! -d ${tfboard_dir} ];
then
    mkdir -p ${tfboard_dir}

fi

if [ ! -d ${model_dir} ];
then
    mkdir -p ${model_dir}

fi

if [ ! -d ${bag_dir} ];
then
    mkdir -p ${bag_dir}

fi


# delete cached redis keys
echo "Clearing redis cached keys ...."
bash clear_redis.sh ${TASK_NAME}



argtasktag="--task=${TASK_NAME}"
arghost="--th=${REDIS_TRAIN_HOST} --ph=${REDIS_PRE_HOST} --mh=${REDIS_MID_HOST} --ch=${REDIS_CON_HOST}"
argport="--tp=${REDIS_TRAIN_PORT} --pp=${REDIS_PRE_PORT} --mp=${REDIS_MID_PORT} --cp=${REDIS_CON_PORT}"
argtfboard_dir="--tfboard_dir=${tfboard_dir}"
argbag_dir="--bag_dir=${bag_dir}"
argpath="--path=${root_dir}"
argmodel_dir="--model_dir=${model_dir}"
# start envs
list_machines=(${ENV_MACHINES})
list_envfiles=(${ENV_FILES})

length=${#list_machines[@]}
echo ${length}
for ((i=0, INDEX_NUM=0; i<${length}; i++));
do
    ENV_MACHINE=${list_machines[$i]}
    ENV_FILE=${list_envfiles[$i]}

    source envs/${ENV_FILE}

    host="drl@${ENV_MACHINE}"
    argtype="--type=env"
    argip="--ip=${ENV_MACHINE}"
    pid_file="${pid_dir}env.pid"
    argenv_dir="--yaml_f=${YAML_F}"

    argtotal="${run_file} ${arghost} ${argport} ${argtype} ${argip} ${argtfboard_dir} ${argpath} ${argtasktag} ${argenv_dir} ${argbag_dir}"
    if [ ${ENV_MACHINE} == "localhost" ]
    then
        echo "nohup python ${argtotal} > ${log_dir}/env.log 2>&1 & echo $! > ${pid_file}"
        nohup python ${argtotal} > ${log_dir}/env.log 2>&1 & echo $! > ${pid_file}
    else
        echo "source ${VENV}; nohup python ${argtotal} > ${log_dir}/env.log 2>&1 & echo $! > ${pid_file}"

        ssh ${host} "source ${VENV}; nohup python ${argtotal} > ${log_dir}/env.log 2>&1 & echo $! > ${pid_file}"
    fi
#    INDEX_NUM=`expr $INDEX_NUM + $PROCESS_NUM`
done


# start trainers
# get env file first, because we need
ENV_FILE=${list_envfiles[0]}
source envs/${ENV_FILE}

for TRA_MACHINE in ${TRI_MACHINES};
do
    host="drl@${TRA_MACHINE}"
    argtype="--type=train"
    pid_file="${pid_dir}train.pid"
    argenv_dir="--yaml_f=${YAML_F}"

    argtotal="${run_file} ${arghost} ${argport} ${argtype} ${argtfboard_dir} ${argpath} ${argmodel_dir} ${argtasktag} ${argenv_dir}"
    if [ ${TRA_MACHINE} == "localhost" ]
    then
        nohup python ${argtotal}> ${log_dir}/train.log 2>&1 & echo $! > ${pid_file}
    else
        ssh ${host} "source ${VENV}; nohup python ${argtotal}> ${log_dir}/train.log 2>&1 & echo $! > ${pid_file}"
    fi
done

# start predictors
for PRE_MACHINE in ${PRE_MACHINES};
do

    host="drl@${PRE_MACHINE}"
    argtype="--type=predict"
    pid_file="${pid_dir}predict.pid"
    argenv_dir="--yaml_f=${YAML_F}"

    argtotal="${run_file} ${arghost} ${argport} ${argtype} ${argtfboard_dir} ${argpath} ${argmodel_dir} ${argtasktag} ${argenv_dir}"
    if [ ${PRE_MACHINE} == "localhost" ]
    then
        nohup python ${argtotal}> ${log_dir}/pre.log 2>&1 & echo $! > ${pid_file}
    else
        ssh ${host} "source ${VENV}; nohup python ${argtotal}> ${log_dir}/pre.log 2>&1 & echo $! > ${pid_file}"
    fi
done



