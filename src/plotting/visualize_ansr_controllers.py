# Run the labyrinth navigation experiment.
''' Conventions

Directions:
----------
0 = right
1 = down
2 = left
3 = up

Actions:
--------
left = 0
right = 1
forward = 2
'''


# %%
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append('..')

from Environments.minigrid_labyrinth import Maze
import numpy as np
from Controllers.minigrid_controller import MiniGridController
from Controllers.meta_controller import MetaController
import pickle
import os, sys
from datetime import datetime
from MDP.high_level_mdp import HLMDP
from utils.results_saver import Results
import time

def obs2waypoint(obs):
    x_minigrid, y_minigrid, yaw_minigrid = obs
    x_cells, y_cells = [25,25]
    x_border, y_border = [2,2]
    x_max, y_max = [128,128]
    
    n_airsim = y_max-(y_minigrid-y_border)*(2*y_max)/(y_cells-2*y_border-1)
    e_airsim = (x_minigrid-x_border)*(2*x_max)/(x_cells-2*x_border-1)-x_max
    
    # Yaw is not yet supported in ROS API
    yaw_airsim = yaw_minigrid
    return ([n_airsim, e_airsim, yaw_airsim])

# %% Setup and create the environment
env_settings = {
    'agent_start_states' : [(22,22,0)],
    'slip_p' : 0.1,
}

env = Maze(**env_settings)

# env.render(highlight=False)
# time.sleep(5)


num_rollouts = 5
meta_controller_n_steps_per_rollout = 500

# %% Set the load directory (if loading pre-trained sub-systems) or create a new directory in which to save results

# load_folder_name = '2023-10-12_13-13-22_minigrid_labyrinth'
load_folder_name = '2024-04-01_17-26-42_minigrid_labyrinth'
save_learned_controllers = True

experiment_name = 'minigrid_labyrinth'

base_path = os.path.abspath(os.path.curdir)
string_ind = base_path.find('src')
assert(string_ind >= 0)
base_path = base_path[0:string_ind + 4]
base_path = os.path.join(base_path, 'ansr_hello_world_py/verifiable-compositional-rl/src/data', 'saved_controllers')

load_dir = os.path.join(base_path, load_folder_name)

# %% Load the sub-system controllers
controller_list = []
for controller_dir in os.listdir(load_dir):
    controller_load_path = os.path.join(load_dir, controller_dir)
    if os.path.isdir(controller_load_path):
        controller = MiniGridController(0, load_dir=controller_load_path)
        controller_list.append(controller)

HL_controller_list = [23,25]

# re-order the controllers by index
reordered_list = []
for HL_controller_id in HL_controller_list:
    for controller in controller_list:
        if controller.controller_ind == HL_controller_id:
            reordered_list.append(controller)
controller_list = reordered_list

obs = env.reset()
print("Init State: "+str(obs))
for controller in controller_list:
    init = True
    if init:
        print("Final State: "+str(controller.get_final_states())+"\n")
        print("** Using Controller **: "+str(controller.controller_ind)+"\n")
        init = False
    while (obs != controller.get_final_states()).any():
        action,_states = controller.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print("Action: "+str(action))
        print("Current State: "+str(obs))
        # env.render(highlight=False)
        # time.sleep(0.5)
        # AirSim mapping
        airsim_obs = obs2waypoint(obs)
        print("AirSim State: "+str(airsim_obs)+"\n")
    