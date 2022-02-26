#!/bin/bash
# source files
source machines/all.sh
source config.sh

file="/home/drl/${USER}/drlnav_frame"
git_host="git@git.ustc.edu.cn:drl_navigation/drlnav_frame.git"


for machine in ${ENV_MACHINES};
do
    host="drl@${machine}"
    echo "$host"
    if ! ssh "$host" "test -e ${file}"
    then
        ssh "$host" "mkdir -p ${file}; cd ${file} ;git clone -b ${USER} ${git_host}; mkdir -p output/"
    fi

    if ssh "$host" "test -e ${file}/output/${TAG}"
    then
        if [ "$1" == "erase" ]
        then
            echo "rm -rf ${file}/output/${TAG}..."
            ssh "$host" "rm -rf ${file}/output/${TAG}"
        else
            echo "${file}/output/${TAG} already exists, you can change version in config.sh"
            exit 8
        fi
    fi
    ssh "$host" "mkdir -p ${file}/output/${TAG}/log ${file}/output/${TAG}/pid ${file}/output/${TAG}/tfboard"
    ssh "$host" "cd ${file}/drlnav_frame; git pull origin ${USER}:${USER}"

    echo "${machine} update code successfully"

done