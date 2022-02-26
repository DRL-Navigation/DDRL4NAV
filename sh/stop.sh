#!/bin/bash
source ./config.sh
source ${REDIS_FILE}
if [ "$1" == "remote" ];
then
    root_dir="/home/drl/${USER}/"
else
    root_dir="`pwd`/../../"
fi
log_dir="${root_dir}${OUTPUT_DIR}${TASK_NAME}/log/"

#TASK_NAME=`head -1 ${log_dir}env.log`
echo "stop $TASK_NAME ..."
redis-cli -h ${REDIS_CON_HOST} -p ${REDIS_CON_PORT} set "${TASK_NAME}EXIT" 1