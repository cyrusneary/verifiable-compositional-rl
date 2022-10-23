# %%
from stable_baselines3 import PPO

import gym
# from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gym_minigrid.minigrid import *

import matplotlib.pyplot as plt

import os, sys
sys.path.append('..')

from Environments.minigrid_pixel_labyrinth import PixelMaze
from Controllers.minigrid_pixel_controller import MiniGridPixelController

env_settings = {
    'agent_start_states' : [(1,1,0)],
    'slip_p' : 0.1,
}

env = PixelMaze(**env_settings)

controller_list = []

initial_states = [(1,1,0)]
final_states = [(3,5,0)]
controller_list.append(MiniGridPixelController(0, initial_states, final_states, env_settings, verbose=True))

initial_states = [(1,1,0)]
final_states = [(5,2,1)]
controller_list.append(MiniGridPixelController(1, initial_states, final_states, env_settings, verbose=True))

# Top long room controllers
initial_states = [(5,2,1)]
final_states = [(10,5,1)]
controller_list.append(MiniGridPixelController(2, initial_states, final_states, env_settings, verbose=True))

initial_states = [(5,2,1)]
final_states = [(14,5,1)]
controller_list.append(MiniGridPixelController(3, initial_states, final_states, env_settings, verbose=True))

# Left room controllers
initial_states = [(3,5,0)]
final_states = [(5,10,1)]
controller_list.append(MiniGridPixelController(4, initial_states, final_states, env_settings, verbose=True))

initial_states = [(5,10,1)]
final_states = [(3,15,1)]
controller_list.append(MiniGridPixelController(5, initial_states, final_states, env_settings, verbose=True))

# Middle room controllers
initial_states = [(10,5,1)]
final_states = [(10,13,1)]
controller_list.append(MiniGridPixelController(6, initial_states, final_states, env_settings, verbose=True))

initial_states = [(10,13,1)]
final_states = [(10,5,3)]
controller_list.append(MiniGridPixelController(7, initial_states, final_states, env_settings, verbose=True))

# Right room controllers
initial_states = [(14,5,1)]
final_states = [(16,15,1)]
controller_list.append(MiniGridPixelController(8, initial_states, final_states, env_settings, verbose=True))

# Bottom room controllers
initial_states = [(3,15,1)]
final_states = env.goal_states
controller_list.append(MiniGridPixelController(9, initial_states, final_states, env_settings, verbose=True))

initial_states = [(16,15,1)]
final_states = [(10,17,2)]
controller_list.append(MiniGridPixelController(10, initial_states, final_states, env_settings, verbose=True))

initial_states = [(10,17,2)]
final_states = env.goal_states
controller_list.append(MiniGridPixelController(11, initial_states, final_states, env_settings, verbose=True))

print(controller_list[4].training_env.slip_p)

# %%
controller_list[4].learn(total_timesteps=100000)


# %%
controller_list[4].eval_performance(n_episodes=400, n_steps=100)

controller_list[4].demonstrate_capabilities(n_episodes=5, n_steps=100, render=True)

# from gym.wrappers.pixel_observation import PixelObservationWrapper
# env = PixelObservationWrapper(env, pixels_only=True)

# # env = RGBImgPartialObsWrapper(env) # Get pixel observations
# # env = ImgObsWrapper(env)     # Get rid of the 'mission' field
# obs = env.reset()

# # img = env.render()

# # plt.imshow(obs)
# # plt.show()

# # img = env.get_obs_render(obs['image'], tile_size=10)
# # plt.imshow(obs['image'])
# # plt.show()

# model = PPO("CnnPolicy", env, verbose=1)
# # model.learn(total_timesteps=25000)

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render(highlight=False)
#     if dones:
#         print(dones)
#         break

