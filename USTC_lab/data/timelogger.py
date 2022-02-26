from USTC_lab.data import HistogramLogger


class TimeLogger(HistogramLogger):
    def __init__(self, **kwargs):
        super(TimeLogger, self).__init__(**kwargs)
        """
        scalar | histogram
        """
        self.tfboard_type = 'histogram'


class BackUpTimeLogger(TimeLogger):
    def __init__(self, **kwargs):
        super(BackUpTimeLogger, self).__init__(**kwargs)
        self.tensor_tag = self.task_name + "{}{}/backuptime"

    def __str__(self):
        return "BackUpTime"


class PpoBackUpTimeLogger(TimeLogger):
    def __init__(self, **kwargs):
        super(PpoBackUpTimeLogger, self).__init__(**kwargs)
        self.tensor_tag = self.task_name + "{}{}/PpoBackUpTime"

    def __str__(self):
        return "PpoBackUpTime"


class GAN_D_BackUpTimeLogger(TimeLogger):
    def __init__(self, **kwargs):
        super(GAN_D_BackUpTimeLogger, self).__init__(**kwargs)
        self.tensor_tag = self.task_name + "{}{}/Gail[D]BackUpTime"

    def __str__(self):
        return "Gail[D]BackUpTime"


class GAN_G_BackUpTimeLogger(TimeLogger):
    def __init__(self, **kwargs):
        super(GAN_G_BackUpTimeLogger, self).__init__(**kwargs)
        self.tensor_tag = self.task_name + "{}{}/Gail[G]BackUpTime"

    def __str__(self):
        return "Gail[G]BackUpTime"


class RNDBackupTimeLogger(TimeLogger):
    def __init__(self, **kwargs):
        super(RNDBackupTimeLogger, self).__init__(**kwargs)
        self.tensor_tag = self.task_name + "{}{}/RndBackUpTime"

    def __str__(self):
        return "RndBackUpTime"




class TrajectoryTimeLogger(TimeLogger):
    def __init__(self, **kwargs):
        super(TrajectoryTimeLogger, self).__init__(**kwargs)

        self.tensor_tags = [self.task_name + "{}{}/traj_time",
                            ]

    def __str__(self):
        return "TrajectoryTime"


class ForwardTimeLogger(TimeLogger):
    def __init__(self, **kwargs):
        super(ForwardTimeLogger, self).__init__(**kwargs)

        self.tensor_tags = [self.task_name + "{}{}/predict_time",
                            ]

    def __str__(self):
        return "ForwardTime-ms"

