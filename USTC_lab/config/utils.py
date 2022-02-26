from typing import List


def startswith_groups(x: str, groups: List):
    for y in groups:
        if x.startswith(y):
            return True
    return False


# def time_limit(gid: str):
#     if startswith_groups(gid, ["Breakout"]):
#         return 4000
#     return None


def game_type(gid: str):
    print(gid)
    if startswith_groups(gid, ['AirRaid', 'AirRaid', 'Alien', 'Alien', 'Amidar', 'Amidar', 'Assault', 'Assault', 'Asterix', 'Asterix', 'Asteroids', 'Asteroids', 'Atlantis', 'Atlantis', 'BankHeist', 'BankHeist', 'BattleZone', 'BattleZone', 'BeamRider', 'BeamRider', 'Berzerk', 'Berzerk', 'Bowling', 'Bowling', 'Boxing', 'Boxing', 'Breakout', 'Breakout', 'Carnival', 'Carnival', 'Centipede', 'Centipede', 'ChopperCommand', 'ChopperCommand', 'CrazyClimber', 'CrazyClimber', 'DemonAttack', 'DemonAttack', 'DoubleDunk', 'DoubleDunk', 'ElevatorAction', 'ElevatorAction', 'Enduro', 'Enduro', 'FishingDerby', 'FishingDerby', 'Freeway', 'Freeway', 'Frostbite', 'Frostbite', 'Gopher', 'Gopher', 'Gravitar', 'Gravitar', 'IceHockey', 'IceHockey', 'Jamesbond', 'Jamesbond', 'JourneyEscape', 'JourneyEscape', 'Kangaroo', 'Kangaroo', 'Krull', 'Krull', 'KungFuMaster', 'KungFuMaster', 'MontezumaRevenge', 'MontezumaRevenge', 'MsPacman', 'MsPacman', 'NameThisGame', 'NameThisGame', 'Phoenix', 'Phoenix', 'Pitfall', 'Pitfall', 'Pong', 'Pong', 'Pooyan', 'Pooyan', 'PrivateEye', 'PrivateEye', 'Qbert', 'Qbert', 'Riverraid', 'Riverraid', 'RoadRunner', 'RoadRunner', 'Robotank', 'Robotank', 'Seaquest', 'Seaquest', 'Skiing', 'Skiing', 'Solaris', 'Solaris', 'SpaceInvaders', 'SpaceInvaders', 'StarGunner', 'StarGunner', 'Tennis', 'Tennis', 'TimePilot', 'TimePilot', 'Tutankham', 'Tutankham', 'UpNDown', 'UpNDown', 'Venture', 'Venture', 'VideoPinball', 'VideoPinball', 'WizardOfWor', 'WizardOfWor', 'YarsRevenge', 'YarsRevenge', 'Zaxxon', 'Zaxxon',]):
        return "atari"
    if startswith_groups(gid, ['Acrobot', 'CartPole', 'MountainCar', 'MountainCarContinuous', 'Pendulum']):
        return "classical"
    if startswith_groups(gid, ['Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'HumanoidStandup', 'InvertedDoublePendulum', 'InvertedPendulum', 'Reacher', 'Swimmer', 'Walker2d']):
        return "mujoco"
    if startswith_groups(gid, ['FetchPickAndPlace', 'FetchPush', 'FetchReach', 'FetchSlide', 'HandManipulateBlock', 'HandManipulateEgg', 'HandManipulatePen', 'HandReach']):
        return "robotics"
    if startswith_groups(gid, ['BipedalWalker', 'BipedalWalkerHardcore', 'CarRacing', 'LunarLander', 'LunarLanderContinuous']):
        return "box2d"
    if startswith_groups(gid, ['Blackjack', 'FrozenLake', 'FrozenLake8x8', 'GuessingGame', 'HotterColder', 'NChain', 'Roulette', 'Taxi']):
        return "toytext"
    if startswith_groups(gid, ['Copy', 'DuplicatedInput', 'RepeatCopy', 'Reverse', 'ReversedAddition', 'ReversedAddition3']):
        return "algorithms"
    if startswith_groups(gid, ["Navigation", "robotnav", "ROBOTnav", "robot_nav"]):
        return "robot_nav"
    raise NameError

