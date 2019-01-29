import gym
import numpy as np
import argparse
from stable_baselines.common.policies import * # MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, 
                                               # CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy 
                                               # ActorCriticPolicy
from stable_baselines.common.vec_env import *  # DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.common import set_global_seeds
from stable_baselines import * # A2C, ACER, ACKTR, DQN, DDPG, PPO1, PPO2, TRPO


class PointEnvBox(gym.Env):
    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

    def __init__(self, ep_length=100):
        set_global_seeds(2020)
        self.ep_length = ep_length
        self.current_step = 0
        self.reset()
    
    def reset(self):
        self.state = np.random.uniform(0, 1, size=(2,))
        observation = np.copy(self.state)
        return observation

    def _get_reward(self):
        x, y = self.state
        reward = 0.0 - np.sqrt(np.square(x) + np.square(y))
        return reward

    def step(self, action):
        self.current_step += 1
        self.state = self.state + action
        
        reward = self._get_reward()
        done = np.abs(reward) < 0.1 #
        # done = abs(x) < 0.1 and abs(y) < 0.1
        if done:
            print('State  :', self.state)
            print('Action :', action)
            print('Rewards:', reward)
        next_observation = np.copy(self.state)
        return next_observation, reward, done, {}



    def render(self, mode='human', close=False):
        print('State  :', self.state)
        print('Rewards:', self._get_reward())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt', default="PointEnvBox_v0", help='load models for continue train or predict')
    parser.add_argument('--train' , action='store_true')
    parser.add_argument('--test' ,  action='store_true')
    parser.add_argument('--seed' ,  action='store', default=2020, type=int)
    args = parser.parse_args() # Create an object of parser

    # Create and wrap the environment
    env = PointEnvBox() 
    env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, 
    #                 norm_obs=True, 
    #                 clip_obs=1.,
    #                 norm_reward=False,
    #                 epsilon=1e-6
    #                 )

    if args.train: # python sb_PointEnvBox.py --train
        # PPO2
        model = PPO2(policy="MlpPolicy", env=env, verbose=1)
        # Train the agent
        model.learn(total_timesteps=20000) 
        # Save the agent
        model.save(args.chkpt)
        # del model  # delete trained model to demonstrate loading

    if args.test: # python sb_PointEnvBox.py --test
        # Load the trained agent
        model = PPO2.load(args.chkpt, env)

        # Test trained agent
        if args.seed:
            set_global_seeds(args.seed)
        else:
            set_global_seeds(2020)
        obs = env.reset()
        env.render()

        for i in range(500):
            act, _ = model.predict(obs, deterministic=True)
            obs, rwd, done, info = env.step(act)
            print('State  :', obs)
            print('Action :', act)
            print('Rewards:', rwd)
            print('Finised:', done)
            # env.render()
            if done:
                break