from USTC_lab.data import ScalarsLogger, ReducedScalarLogger


class RateEpisodeLogger(ScalarsLogger):
    def __init__(self, **kwargs):
        super(RateEpisodeLogger, self).__init__(**kwargs)


class ReachRateEpisodeLogger(RateEpisodeLogger):
    def __init__(self, **kwargs):
        super(ReachRateEpisodeLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReachRateEpisode"


class StaticObsCollisionRateEpisode(RateEpisodeLogger):
    def __init__(self, **kwargs):
        super(StaticObsCollisionRateEpisode, self).__init__(**kwargs)

    def __str__(self):
        return "StaticObsCollisionRateEpisode"


class PedCollisionRateEpisode(RateEpisodeLogger):
    def __init__(self, **kwargs):
        super(PedCollisionRateEpisode, self).__init__(**kwargs)

    def __str__(self):
        return "PedCollisionRateEpisode"


class OtherRobotCollisionRateEpisode(RateEpisodeLogger):
    def __init__(self, **kwargs):
        super(OtherRobotCollisionRateEpisode, self).__init__(**kwargs)

    def __str__(self):
        return "OtherRobotCollisionRateEpisode"


class ReducedReachRateLogger(ReducedScalarLogger, RateEpisodeLogger):
    def __init__(self, **kwargs):
        super(ReducedReachRateLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedReachRateLogger"


class ReducedStaticObsCollisionRateLogger(ReducedScalarLogger, RateEpisodeLogger):
    def __init__(self, **kwargs):
        super(ReducedStaticObsCollisionRateLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedStaticObsCollisionRateLogger"


class ReducedPedCollisionRateLogger(ReducedScalarLogger, RateEpisodeLogger):
    def __init__(self, **kwargs):
        super(ReducedPedCollisionRateLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedPedCollisionRateLogger"


class ReducedOtherRobotCollisionRateLogger(ReducedScalarLogger, RateEpisodeLogger):
    def __init__(self, **kwargs):
        super(ReducedOtherRobotCollisionRateLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ReducedOtherRobotCollisionRateLogger"
