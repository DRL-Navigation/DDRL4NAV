""" dynamic change [agents | trainers | predictors] during the running
"""
#  TODO change to shell script


from USTC_lab.config import Config
import redis

agents, trainers, predictors = list(map(int, input("Please input nums of [agents | trainers | predictors]").split()))

conn = redis.Redis(Config.CONTROL_REDIS_HOST, Config.CONTROL_REDIS_PORT)
pipe = conn.pipeline()
pipe.set(Config.TASK_NAME + Config.ENV_NUM_KEY, agents)
pipe.set(Config.TASK_NAME + Config.TRAINERS_NUM_KEY, trainers)
pipe.set(Config.TASK_NAME + Config.PREDICTORS_NUM_KEY, predictors)

pipe.execute()

print("set {} agents, {} trains, {} predictors successfully! ".format(agents, trainers, predictors))