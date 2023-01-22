"""Wrapper that stacks frames."""
from collections import deque, OrderedDict
from typing import Union

import numpy as np
import robomimic.envs.env_robosuite as EB

class FrameStack(EnvRobosuite):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Note:
        - To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
        - The observation space must be :class:`Box` type. If one uses :class:`Dict`
          as observation space, it should apply :class:`FlattenObservation` wrapper first.
          - After :meth:`reset` is called, the frame buffer will be filled with the initial observation. I.e. the observation returned by :meth:`reset` will consist of ``num_stack`-many identical frames,

    """

    def __init__(
        self,
        env,
        num_stack: int,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        super().__init__(env)
        self.num_stack = num_stack

        self.frames = deque(maxlen=num_stack)


    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        obs = OrderedDict()
        for key in self.frames[0].keys():
            obs[key] = np.stack([self.frames[ind][key] for ind in range(self.num_stack)], axis=0)

        return obs

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(None), reward, terminated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs = self.env.reset(**kwargs)

        [self.frames.append(obs) for _ in range(self.num_stack)]

        return self.observation(None)
