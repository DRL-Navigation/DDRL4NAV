import gym


class FilterStatesWrapper(gym.ObservationWrapper):
    def __init__(self, env, cfg):
        super(FilterStatesWrapper, self).__init__(env)

    def observation(self, states):

        return [states]
