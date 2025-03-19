import numpy as np
import gymnasium as gym
import bsk_envs

env = gym.make('OrbitDiscovery3DOF-v0', render_mode='human')

done = False
state, _ = env.reset()
total_reward = 0
while not done:
    state, reward, done, _, info = env.step(env.action_space.sample())
    total_reward += reward
    env.render()
env.close()



