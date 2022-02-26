import numpy as np
import gym
import torch
import numpy as np
import cv2

from collections import deque

from numbers import Number
from typing import List, Union
from ..wrapper import dtype_dict


def start_score(name: str) -> float:
    if name.startswith("Pong"):
        return -21.0

    return 0.0



"""
    from tianshou https://github.com/thu-ml/tianshou/blob/ebaca6f8da91e18e0192184c24f5d13e3a5d0092/tianshou/utils/statistics.py
"""
class MovAvg(object):
    """Class for moving average.
    It will automatically exclude the infinity and NaN. Usage:
    ::
        >>> stat = MovAvg(size=66)
        >>> stat.add(torch.tensor(5))
        5.0
        >>> stat.add(float('inf'))  # which will not add to stat
        5.0
        >>> stat.add([6, 7, 8])
        6.5
        >>> stat.get()
        6.5
        >>> print(f'{stat.mean():.2f}±{stat.std():.2f}')
        6.50±1.12
    """

    def __init__(self, size: int = 100) -> None:
        super().__init__()
        self.size = size
        self.cache: List[np.number] = []
        self.banned = [np.inf, np.nan, -np.inf]

    def add(
        self, x: Union[Number, np.number, list, np.ndarray, torch.Tensor]
    ) -> float:
        """Add a scalar into :class:`MovAvg`.
        You can add ``torch.Tensor`` with only one element, a python scalar, or
        a list of python scalar.
        """
        if isinstance(x, torch.Tensor):
            x = x.flatten().cpu().numpy()
        if np.isscalar(x):
            x = [x]
        for i in x:  # type: ignore
            if i not in self.banned:
                self.cache.append(i)
        if self.size > 0 and len(self.cache) > self.size:
            self.cache = self.cache[-self.size:]
        return self.get()

    def get(self) -> float:
        """Get the average."""
        if len(self.cache) == 0:
            return 0.0
        return float(np.mean(self.cache))

    def mean(self) -> float:
        """Get the average. Same as :meth:`get`."""
        return self.get()

    def std(self) -> float:
        """Get the standard deviation."""
        if len(self.cache) == 0:
            return 0.0
        return float(np.std(self.cache))


class RunningMeanStd(object):
    """Calulates the running mean and std of a data stream.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(
        self, mean: Union[float, np.ndarray] = 0.0, std: Union[float, np.ndarray] = 1.0
    ) -> None:
        self.mean, self.var = mean, std
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(x, axis=0), np.var(x, axis=0)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(FrameStackWrapper, self).__init__(env)
        self.k = cfg['int_frame_stack']
        self.q = deque([], maxlen=self.k)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.q.append(observation)
        return self.observation, reward, done, info

    @property
    def observation(self):
        return np.concatenate(list(self.q), axis=1)

    def reset(self):
        obs = self.env.reset()
        for i in range(self.k):
            self.q.append(obs)
        return self.observation


class ExpandWrapper(gym.ObservationWrapper):
    def __init__(self, env, cfg):
        super(ExpandWrapper, self).__init__(env)

    def observation(self, observation):
        return observation[None]


class MultiWarp(gym.Wrapper):
    def __init__(self, env):
        super(MultiWarp, self).__init__(env)

    def step(self, action: np.ndarray):
        obs, r, done, _ = self.env.step(self.action(action))
        return obs[None].astype(np.float32), np.array([r], dtype=np.float32), np.array([1 if done else 0], dtype=np.float32), _

    def action(self, action):
        if action.dtype == np.uint8:
            return int(action[0])
        else:
            return action

    def reset(self, **kwargs):
        return self.env.reset()[None].astype(np.float32)



class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class WarpEpisodeLife(gym.Wrapper):
    def __init__(self, env, cfg):
        super(WarpEpisodeLife, self).__init__(env)
        self.lives: int = cfg['int_lives']
        self.real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.real_done = done
        # check current lives
        lives = self.env.unwrapped.ale.lives()
        # if died
        if lives > 0 and lives < self.lives:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)

        self.lives = self.env.unwrapped.ale.lives()
        return obs


class WarpFrameWrapper(gym.ObservationWrapper):
    def __init__(self, env, cfg):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = 84
        self._height = 84
        self._grayscale = True
        self._key = None
        self.i=0
        assert cfg["state_dtype"] in dtype_dict.keys()

        self.nptype = dtype_dict[cfg["state_dtype"]]
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=self.nptype,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame[34:194], (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, 0)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        # print(obs.shape)
        # from PIL import Image
        #
        # im = Image.fromarray(obs[0])
        # im.save('{}.jpg'.format(self.i))
        # self.i += 1

        obs = obs.astype(self.nptype) / 255.0
        return obs[None]


# class WarpAction(gym.ActionWrapper):
#     def __init__(self, env):
#         super(WarpAction, self).__init__(env)
#
#     def action(self, action):
#         return action
class WrapAction(gym.ActionWrapper):
    def __init__(self, env, cfg):
        super(WrapAction, self).__init__(env)
        self.discrete = cfg['discrete_action']

    def action(self, action):
        if self.discrete:
            action = int(action)

        return action



class WarpReward(gym.Wrapper):

    def __init__(self, env, cfg):
        super(WarpReward, self).__init__(env)

        assert cfg["reward_dtype"] in dtype_dict.keys()
        self.dtype = dtype_dict[cfg["reward_dtype"]]
        self.reward_clip: bool = cfg['bool_reward_clip']
        self.reward_min: float = cfg['float_min_reward']
        self.reward_max: float = cfg['float_max_reward']

        # self.reward_die: float = cfg['float_die_reward']

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info["origin_r"] = np.array([reward], dtype=self.dtype)
        return observation, self.reward(reward, done), done, info

    def reward(self, reward, done):
        # if done:
        #     assert done is bool or int
        #     reward = self.reward_die
        if self.reward_clip:
            reward = np.clip(reward, self.reward_min, self.reward_max)

        return np.array([reward], dtype=self.dtype)



class WarpDone(gym.Wrapper):
    def __init__(self, env, cfg):
        super(WarpDone, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, self.done(done), info

    def done(self, done):
        done = 1 if done else 0
        return np.array([done], dtype=np.float32)



# time limit
class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(TimeLimitWrapper, self).__init__(env)
        self._max_episode_steps = cfg['time_max']
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        done = np.where(self._elapsed_steps > self._max_episode_steps, 100, done)
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)###


class NeverStopWrapper(gym.Wrapper):
    """
        NOTE !!!!!!!!!!!
        put this in last wrapper.
    """
    def __init__(self, env, cfg):
        super(NeverStopWrapper, self).__init__(env)

    def step(self, action):
        states, reward, done, info = self.env.step(action)
        if sum(np.where(done>0, 1, 0)) == len(done):
            states = self.env.reset()

        return states, reward, done, info


class DisplayWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(DisplayWrapper, self).__init__(env)
        self.display = cfg['show_gui']

    def step(self, action):
        if self.display:
            self.env.render()
        return self.env.step(action)


class InfoExpandWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(InfoExpandWrapper, self).__init__(env)

    def step(self, action):
        states, reward, done, info = self.env.step(action)
        info['lives'] = np.array([info['lives']], dtype=np.uint8)
        info['all_down'] = done
        info['is_clean'] = np.array([True])
        return states, reward, done, info