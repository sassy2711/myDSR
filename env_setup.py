import gymnasium as gym
import numpy as np
import mujoco

def make_env(env_name):
    env = gym.make(env_name)
    return env

if __name__ == "__main__":
    env = make_env("Reacher-v5")  # Updated to the latest version
    obs, info = env.reset()  # Updated API: reset() returns (obs, info)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    #print(env.action_space.shape[0])
