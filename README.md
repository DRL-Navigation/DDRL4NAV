# Distributed DRL Frame For Robot Navigation

### Introduction

We reproduce a light RL training framework from [OpenAi Five](https://arxiv.org/abs/1912.06680). As seen in the following, the structure of our framework is totally the same as the paper shown. 3 key ingredients in the RL training process, Forward Module, Backward Module, and Env Module are separated.

![image-20220128203623248](png/README.assets/image-20220128203623248.png)



### ENVIRONMENT SETTING

recommend os : centos | ubuntu16.04+

initialize your working directory in the beginning

```
chsh -s /bin/bash    # make sure you are in a bash-based terminal
sudo mkdir -p  /home/${USER}/drlnav_frame
```

#### Redis

**if you have not installed redis-6.x yet,  please type the following command in your terminal** 

```
cd /home/${USER}/drlnav_frame
git clone git@git.ustc.edu.cn:drl_navigation/drlnav_frame-tools.git
cd drlnav_frame-tools
sudo tar -xzvf redis/redis-6.2.1.tar.gz -C /usr/local/
sudo ln -s /usr/local/redis-6.2.1/src/redis-server /usr/bin/redis-server
sudo apt install redis-tools
```

#### Python3.8

**if you have not installed python3.7+ yet,  please type the following command in your terminal**

**Otherwise, goto the third command directly**

**first**: we need to fix the **_ctype** error for python3.7 +

ubuntu:

```py
sudo apt-get install libffi-dev
```

centos

```py
yum install libffi-devel
```
if you are in ubuntu20.04, pass second.

**second** :configure python3.8

```
cd drlnav_frame-tools
sudo tar -xzvf python3/Python-3.8.4.tgz -C /usr/local/
cd /usr/local/Python-3.8.4
./configure
make -j4
sudo make install
```

**third**: create venv

```
cd /home/${USER}/drlnav_frame
pip3 install --user virtualenv
python3.8 -m virtualenv venv38

```

**forth**: source venv

```
source venv38/bin/activate
pip install -r requirements.txt
# then install pytorch: 
# see https://pytorch.org/get-started/locally/
# pip+cenos+cuda11.3:
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

if you see (venv38)  in the head of your terminal , it means installed venv successfully!
```py
python
>>>import torch
>>>torch.cuda.is_available()
True
```
Notion: if you see False in your python console, just fit in pytorch version with your cuda .[Torch Version](https://pytorch.org/get-started/previous-versions)

now , the whole project dir tree like this:

```sh
-drlnav_frame
​    -drlnav_frame-tools
​         -redis
​         -python3	
​         -README.md
​    -venv38
```

### Quick Start

supposed you are in /home/${USER}/drlnav_frame dir

```
git clone git@git.ustc.edu.cn:drl_navigation/drlnav_frame.git --recurse-submodules
```

now , the whole project dir tree like this:

```sh
-drlnav_frame
​    -drlnav_frame
​         -USTC_lab
​         -sh
​         -requirements.txt
​         -README.md
​    -drlnav_frame-tools
​         -redis
​         -python3	
​         -README.md
​    -venv38
```

now, let's try to train one classical atari game - Pong in your local machine

If you want to run robot navigation, you should also look [nav_env guide](https://git.ustc.edu.cn/drl_navigation/drlnav_env) to run ros node.

**Notion: make sure to check USER in sh/config.sh**

```
cd /home/${USER}/drlnav_frame/drlnav_frame/sh
bash start_redis.sh
bash start.sh
```

open tensorboard page :

```
bash tfboard.sh 
```

stop training

```
# warmly stop
bash stop.sh
# or you can just kill them
bash kill_all.sh
```
### Connection between Env and Net
For training a neural network, you have to pick some kinds of states which observed by your env.
In addition, you have to define encoding network to encode states, and put the encoded states to AC,
Notice that `LAST_INPUT_DIM dim` in `config_nn.py`  means the dim of encoded states which will flow into
Actor and Critic later.


### Distributed Training Start

**first**: connect to [Jump Server](https://zh.wikipedia.org/wiki/%E8%B7%B3%E6%9D%BF%E6%9C%BA)

```

```

**second**:  update repo

you should have modify personal branch **in your working machine**

```
git clone git@git.ustc.edu.cn:drl_navigation/drlnav_frame.git
git checkout -b qiuqc

deploy config.sh

modify sh/machines/all.sh

modify sh/envs/XX.sh  # XX.sh setting in sh/config.sh

git push origin qiuqc:qiuqc
```

**In Jump Server**

first time 

```
USER=qiuqc # Notion: input your name here
mkdir -p /home/drl/{USER}/drlnav_frame; cd /home/drl/${USER}/drlnav_frame
git clone -b ${USER} git@git.ustc.edu.cn:drl_navigation/drlnav_frame.git
```

then execute pull.sh to pull latest code in workers

```
cd /home/drl/${USER}/drlnav_frame/drlnav_frame/sh
bash pull.sh
```

**third**: start

```
cd /home/drl/${USER}/drlnav_frame/drlnav_frame/sh
bash start_redis.sh
bash start.sh remote
```

finally, open tensorboard then stop training if necessary

```
bash tfboard.sh remote
bash stop.sh remote
```
Jump Server file tree
```sh
-home
  -drl
    -UserA
      -drlnav_frame
        -drlnav_frame
        -output
    -UserB
      -drlnav_frame
        -drlnav_frame
        -output
```

### [Jump Server](https://zh.wikipedia.org/wiki/%E8%B7%B3%E6%9D%BF%E6%9C%BA)/跳板机

- if you want to train in a distributed way, you should always connect to Jump Server before doing anything.

- you should **git push** your personal branch before executing **bash pull.sh remote** in Jump Server

  

### Supplement

- Supports navigation tasks in different machines running in different environments, while using the same network for training. Note that currently only supports n processes running with same environment in one machine(worker). If necessary, this function can also be optimized into a machine with n processes running in n environments.




### Other
- about git submodule:  [Git中submodule的使用](https://zhuanlan.zhihu.com/p/87053283)