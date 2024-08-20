
# Run the labyrinth navigation experiment.

# %%
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append('..')

from Environments.minigrid_pixel_labyrinth import PixelMaze
import numpy as np
from Controllers.minigrid_pixel_controller import MiniGridPixelController
from Controllers.pixel_meta_controller import PixelMetaController
import pickle
import os, sys
from datetime import datetime
from MDP.high_level_mdp import HLMDP
from utils.results_saver import Results

# %% Setup and create the environment
env_settings = {
    'agent_start_states' : [(1,1,0)],
    'slip_p' : 0.2,
}

env = PixelMaze(**env_settings)

num_rollouts = 5
meta_controller_n_steps_per_rollout = 500

# %% Set the load directory (if loading pre-trained sub-systems) or create a new directory in which to save results

#########################################
# Argument command window prompt 


# Set up argument parser
parser = argparse.ArgumentParser(description='Please provide the folder name and experiment name.')
#parser.add_argument('experiment_name', type=str, help='Name of the experiment')
parser.add_argument('load_folder_name', type=str, help='Name of the folder to load')


# Parse arguments
args = parser.parse_args()

# Use the command-line arguments
#experiment_name = args.experiment_name
load_folder_name = args.load_folder_name



########################################


#load_folder_name = '2022-10-13_21-48-57_minigrid_pixel_labyrinth'
# load_folder_name = '2022-10-13_22-25-00_minigrid_pixel_labyrinth'
save_learned_controllers = True

#experiment_name = 'minigrid_pixel_labyrinth'

base_path = os.path.abspath(os.path.curdir)
string_ind = base_path.find('src')
assert(string_ind >= 0)
base_path = base_path[0:string_ind + 4]
base_path = os.path.join(base_path, 'data', 'saved_controllers')

load_dir = os.path.join(base_path, load_folder_name)

# %% Load the sub-system controllers
controller_list = []
for controller_dir in os.listdir(load_dir):
    controller_load_path = os.path.join(load_dir, controller_dir)
    if os.path.isdir(controller_load_path):
        controller = MiniGridPixelController(0, load_dir=controller_load_path)
        controller_list.append(controller)

# re-order the controllers by index
reordered_list = []
for i in range(len(controller_list)):
    for controller in controller_list:
        if controller.controller_ind == i:
            reordered_list.append(controller)
controller_list = reordered_list

# Construct high-level MDP and solve for the max reach probability
hlmdp = HLMDP([(1,1,0)], env.goal_states, controller_list)
policy, reach_prob, feasible_flag = hlmdp.solve_max_reach_prob_policy()

# Construct a meta-controller and emprirically evaluate it.
meta_controller = PixelMetaController(policy, hlmdp.controller_list, hlmdp.state_list)

# %%
meta_success_rate = meta_controller.demonstrate_capabilities(env, n_episodes=num_rollouts, n_steps=meta_controller_n_steps_per_rollout)
# %%
