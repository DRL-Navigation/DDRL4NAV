from USTC_lab.manager.manager import Manager
from USTC_lab.manager.agents_manager import AgentsManager
from USTC_lab.manager.predictor_manager import PredictorManager
from USTC_lab.manager.trainer_manager import TrainerManager



def ManagerFactory(manager_type: str):
    assert manager_type in ["env", "train", "predict"]
    if manager_type == "env":
        return AgentsManager
    if manager_type == "train":
        return TrainerManager
    if manager_type == "predict":
        return PredictorManager




__all__ = [
    "Manager",
    "AgentsManager",
    "PredictorManager",
    "TrainerManager",
    "manager",
    "ManagerFactory"


]