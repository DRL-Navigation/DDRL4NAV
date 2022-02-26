import yaml
import gym

try:
    from USTC_lab.env import make_gymenv
except:
    import sys
    print("Guess your terminal is in gym_env dir, ")
    sys.path.append("../../../")
    from USTC_lab.env import make_gymenv

def read_yaml(file: str) -> dict:
    try:
        file = open(file, 'r', encoding="utf-8")
        # 读取文件中的所有数据
        file_data = file.read()
        file.close()
        # 指定Loader
        dict_original = yaml.load(file_data, Loader=yaml.FullLoader)
        update_cfg_env(dict_original)
        return dict_original
    except:
        return {}

def update_cfg_env(cfg):
    # used for config_nn
    if cfg['env_type'] == 'gym':
        env = gym.make(cfg['env_name'])

        if type(env.action_space) == gym.spaces.Discrete:
            cfg['discrete_action'] = True
            cfg['discrete_actions'] = list(range(env.action_space.n))
        elif type(env.action_space) == gym.spaces.Box:
            cfg['discrete_action'] = False
            # TODO support more than one dim
            cfg['act_dim'] = env.action_space.shape[0]


if __name__ == "__main__":
    dict_cfg = read_yaml("cfg/atari.yaml")
    env = make_gymenv(dict_cfg)
    s = env.reset()
    # print(s.shape)

    while True:
        a,b,c,d = env.step(0)
    #     print(a,b,c,d)
    #     print(b,c,d)
    #     env.render()