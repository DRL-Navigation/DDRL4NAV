from USTC_lab.data import ScalarsLogger, ReducedScalarLogger

"""
    define all RewardLogger in tensorboard here
"""


class ReducedRewardLogger(ReducedScalarLogger):
    """reduced reward group by multi environments,
       such as ppo paper shows"""
    def __init__(self, **kwargs):
        super(ReducedRewardLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedRewardLogger"


class RewardEpisodeLogger(ScalarsLogger):
    def __init__(self, **kwargs):
        super(RewardEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "RewardEpisode"


class DRewardEpisodeLogger(ScalarsLogger):
    def __init__(self, **kwargs):
        super(DRewardEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "D_RewardEpisode"


class DRewardStepLogger(ScalarsLogger):
    def __init__(self, **kwargs):
        super(DRewardStepLogger, self).__init__(**kwargs)

    def __str__(self):
        return "GAN[D]_RewardStep"


class RndRewardStepLogger(ScalarsLogger):
    def __init__(self, **kwargs):
        super(RndRewardStepLogger, self).__init__(**kwargs)

    def __str__(self):
        return "Rnd_RewardStep"



