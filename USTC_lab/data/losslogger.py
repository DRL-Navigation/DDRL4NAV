from USTC_lab.data import ScalarLogger

"""
    define all LossLogger in tensorboard here
"""

"""
    PPO Loss
"""


class LossLogger(ScalarLogger):
    def __init__(self, **kwargs):
        super(LossLogger, self).__init__(**kwargs)

        self.tfboard_type = 'scalar'

    def __str__(self):
        return "Loss"


class TotalLossLogger(LossLogger):
    def __init__(self, **kwargs):
        super(TotalLossLogger, self).__init__(**kwargs)

    def __str__(self):
        return "TotalLoss"


class PpoTotalLossLogger(LossLogger):
    def __init__(self, **kwargs):
        super(PpoTotalLossLogger, self).__init__(**kwargs)

    def __str__(self):
        return "PpoTotalLoss"


class ActorLossLogger(LossLogger):
    def __init__(self, **kwargs):
        super(ActorLossLogger, self).__init__(**kwargs)

    def __str__(self):
        return "ActorLoss"


class VLossLogger(LossLogger):
    def __init__(self, **kwargs):
        super(VLossLogger, self).__init__(**kwargs)

    def __str__(self):
        return "VLoss"


class EntLossLogger(LossLogger):
    def __init__(self, **kwargs):
        super(EntLossLogger, self).__init__(**kwargs)

    def __str__(self):
        return "EntLoss"






"""
    GAN Loss
"""


class GAN_D_LossLogger(LossLogger):
    def __init__(self, **kwargs):
        super(GAN_D_LossLogger, self).__init__(**kwargs)

    def __str__(self):
        return "Gail[D]Loss"


"""
    RND Loss
"""


class RNDLossLogger(LossLogger):
    def __init__(self, **kwargs):
        super(RNDLossLogger, self).__init__(**kwargs)

    def __str__(self):
        return "RndLoss"