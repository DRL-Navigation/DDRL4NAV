from USTC_lab.data import ScalarsLogger, ReducedScalarLogger
"""
speed 
"""
class VelocityEpisodeLogger(ScalarsLogger):
    def __init__(self, **kwargs):
        super(VelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "VelocityEpisodeLogger"


class OpenAreaLinearVelocityEpisodeLogger(VelocityEpisodeLogger):
    def __init__(self, **kwargs):
        super(OpenAreaLinearVelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "OpenAreaLinearVelocityEpisodeLogger"


class OpenAreaAngularVelocityEpisodeLogger(VelocityEpisodeLogger):
    def __init__(self, **kwargs):
        super(OpenAreaAngularVelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "OpenAreaAngularVelocityEpisodeLogger"


class CloseHumanAngularVelocityEpisodeLogger(VelocityEpisodeLogger):
    def __init__(self, **kwargs):
        super(CloseHumanAngularVelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "CloseHumanAngularVelocityEpisodeLogger"


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


class CloseHumanLinearVelocityEpisodeLogger(VelocityEpisodeLogger):
    def __init__(self, **kwargs):
        super(CloseHumanLinearVelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "CloseHumanLinearVelocityEpisodeLogger"


class ReducedCloseHumanLinearVelocityEpisodeLogger(ReducedScalarLogger):
    """reduced LinearVelocity group by multi environments,
       such as ppo paper shows"""
    def __init__(self, **kwargs):
        super(ReducedCloseHumanLinearVelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedCloseHumanLinearVelocityEpisodeLogger"


class ReducedCloseHumanAngularVelocityEpisodeLogger(ReducedScalarLogger):
    """reduced AngularVelocity group by multi environments,
       such as ppo paper shows"""
    def __init__(self, **kwargs):
        super(ReducedCloseHumanAngularVelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedCloseHumanAngularVelocityEpisodeLogger"


class ReducedOpenAreaLinearVelocityEpisodeLogger(ReducedScalarLogger):
    """reduced LinearVelocity group by multi environments,
       such as ppo paper shows"""
    def __init__(self, **kwargs):
        super(ReducedOpenAreaLinearVelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedOpenAreaLinearVelocityEpisodeLogger"


class ReducedOpenAreaAngularVelocityEpisodeLogger(ReducedScalarLogger):
    """reduced AngularVelocity group by multi environments,
       such as ppo paper shows"""
    def __init__(self, **kwargs):
        super(ReducedOpenAreaAngularVelocityEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedOpenAreaAngularVelocityEpisodeLogger"


class ReducedStepsEpisodeLogger(ReducedScalarLogger):
    """reduced trajectory step group by multi environments,
       such as ppo paper shows"""
    def __init__(self, **kwargs):
        super(ReducedStepsEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedStepsEpisodeLogger"