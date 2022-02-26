from USTC_lab.data import ScalarsLogger, ReducedScalarLogger
"""
speed 
"""
class VelocityEpisodeLogger(ScalarsLogger):
    def __init__(self, **kwargs):
        super(VelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "VelocityEpisodeLogger"


class LinearVelocityEpisodeLogger(VelocityEpisodeLogger):
    def __init__(self, **kwargs):
        super(LinearVelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "LinearVelocityEpisodeLogger"


class AngularVelocityEpisodeLogger(VelocityEpisodeLogger):
    def __init__(self, **kwargs):
        super(AngularVelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "AngularVelocityEpisodeLogger"


class ReducedLinearVelocityEpisodeLogger(ReducedScalarLogger):
    """reduced LinearVelocity group by multi environments,
       such as ppo paper shows"""
    def __init__(self, **kwargs):
        super(ReducedLinearVelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedLinearVelocityEpisodeLogger"


class ReducedAngularVelocityEpisodeLogger(ReducedScalarLogger):
    """reduced AngularVelocity group by multi environments,
       such as ppo paper shows"""
    def __init__(self, **kwargs):
        super(ReducedAngularVelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedAngularVelocityEpisodeLogger"


class ReducedStepsEpisodeLogger(ReducedScalarLogger):
    """reduced trajectory step group by multi environments,
       such as ppo paper shows"""
    def __init__(self, **kwargs):
        super(ReducedStepsEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedStepsEpisodeLogger"