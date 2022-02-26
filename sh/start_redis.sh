#!/bin/bash


source redisc/hosts.sh
remote_file="/home/drl/${USER}/USTC_lab/sh/redisc"

hosts=(${REDIS_TRAIN_HOST} ${REDIS_PRE_HOST} ${REDIS_MID_HOST} ${REDIS_CON_HOST})
ports=(${REDIS_TRAIN_PORT} ${REDIS_PRE_PORT} ${REDIS_MID_PORT} ${REDIS_CON_PORT})


for ((i=0; i<4; i++));
do
    host=${hosts[$i]}
    port=${ports[$i]}
    echo "start redis-server -h ${host} -p ${port}"
    if [ ${host} == "localhost" ];
    then
        count=`ps -ef |grep ":${port}" |grep -v "grep" | wc -l`
        echo $count
        if [ 0 == $count ];
        then
            cf="redisc/redis_${port}.conf"
            echo `pwd`
            cp redisc/redis.conf ${cf} ;echo "port ${port}" >> ${cf}
            nohup redis-server ${cf} >/dev/null 2>&1 &
        fi
    else
        train_host="root@${host}"
        count=`ssh ${train_host} "ps -ef |grep :${port} |grep -v grep | wc -l"`
        if [ 0 == $count ];
        then
            cf="redis_${port}.conf"
            ssh $train_host "cd ${remote_file}; cp redis.conf ${cf} ;echo "port ${port}" >> ${cf}"
            ssh $train_host "nohup redis-server ${cf} >/dev/null 2>&1 &"
        fi
    fi
done
