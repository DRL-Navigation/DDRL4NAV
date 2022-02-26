import redis
from USTC_lab.config import Config

#  TODO change to shell script

conn = redis.Redis(Config.CONTROL_REDIS_HOST, Config.CONTROL_REDIS_PORT)
pipe = conn.pipeline()
print("You can choose to show or close gym gui \nsuch as :")
print("    show 0 1 2 3")
print("    close 0 1 2")
ls = input("Please input:")

inp = ls.split()
if inp[0] == 'show':
    for i in inp[1:]:
        pipe.set(Config.RENDER_KEY.format(i), 1)
    pipe.execute()
elif inp[0] == 'close':
    for i in inp[1:]:
        pipe.set(Config.RENDER_KEY.format(i), 0)
    pipe.execute()
else:
    print("Please input the right order!")
