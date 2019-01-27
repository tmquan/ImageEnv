
## Customize Base Processing Environment for Reinforcement Learning

This repository collects several ways of customize the environment for Reinsforcement Learning study. 

- [x] BaseEnv
- [ ] PointEnv
- [ ] ImageEnv
- [ ] More to come

----------
A common approach is to manually register the environment class that was inherited from `gym.Env`

Typical directory structure is as follows

```bash
├── CustomEnv
│   ├── envs
│   │   ├── BaseEnv.py
│   │   └── __init__.py
│   └── __init__.py
├── make_BaseEnv_plain.py

```

The content of `CustomEnv/__init__.py` is
```python
from gym.envs.registration import register

register(
    id='BaseEnv-v0', 					  # Modify here
    entry_point='CustomEnv.envs:BaseEnv', # Modify here
)

# New environments should be registered after here
```


The content of `CustomEnv/envs/__init__.py` is
```python
from CustomEnv.envs.BaseEnv import * # Modify here
```


The content of `CustomEnv/envs/BaseEnv.py` is
```python
import gym

class BaseEnv(gym.Env): # Modify here
    """
    A template to implement custom OpenAI Gym environments
    
    """

    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.__version__ = "0.0.1"
        print("Initialized BaseEnv") # Modify here
        # Modify the observation space, low, high and shape values according to your custom environment's needs
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))
        # Modify the action space, and dimension according to your custom environment's needs
        # self.action_space = gym.spaces.Discrete(4)
        pass

    def step(self, action):
        """
        Runs one time-step of the environment's dynamics. The reset() method is called at the end of every episode
        :param action: The action to be executed in the environment
        :return: (observation, reward, done, info)
            observation (object):
                Observation from the environment at the current time-step
            reward (float):
                Reward from the environment due to the previous action performed
            done (bool):
                a boolean, indicating whether the episode has ended
            info (dict):
                a dictionary containing additional information about the previous action
        """
        # Implement your step method here
        # return (observation, reward, done, info)
        pass

    def reset(self):
        """
        Reset the environment state and returns an initial observation

        Returns
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """

        # Implement your reset method here
        # return observation
        pass

    def render(self, mode='human', close=False):
        """

        :param mode:
        :return:
        """
        pass
```

It should print this line 
```bash
Initialized BaseEnv
```

----------
Another way is to install `rllab` and implement the new environment by wrapping `rllab.envs.base.Env`

```bash
pip3 install theano
pip3 install cached_property
pip3 install git+https://github.com/rll/rllab
```

The content of `make_BaseEnv_rllab.py` is
```python
from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np
import gym

class BaseEnv(Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.__version__ = "0.0.1"
        print("Initialized BaseEnv") # Modify here
        # Modify the observation space, low, high and shape values according to your custom environment's needs
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))
        # Modify the action space, and dimension according to your custom environment's needs
        # self.action_space = gym.spaces.Discrete(4)
        pass

    def step(self, action):
        """
        Runs one time-step of the environment's dynamics. The reset() method is called at the end of every episode
        :param action: The action to be executed in the environment
        :return: (observation, reward, done, info)
            observation (object):
                Observation from the environment at the current time-step
            reward (float):
                Reward from the environment due to the previous action performed
            done (bool):
                a boolean, indicating whether the episode has ended
            info (dict):
                a dictionary containing additional information about the previous action
        """
        # Implement your step method here
        # return (observation, reward, done, info)
        pass

    def reset(self):
        """
        Reset the environment state and returns an initial observation

        Returns
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """

        # Implement your reset method here
        # return observation
        pass

    def render(self, mode='human', close=False):
        """

        :param mode:
        :return:
        """
        pass

if __name__ == '__main__':
    env = BaseEnv()
```

It should print this line 
```bash
Initialized BaseEnv
```

## Basic markdown syntax can be found here 
[https://guides.github.com/features/mastering-markdown/]()
