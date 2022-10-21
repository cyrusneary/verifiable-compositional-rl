
# Run the labyrinth navigation experiment.

# %%
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append('..')

from Environments.unity_labyrinth import build_unity_labyrinth_env
import numpy as np
from Controllers.unity_labyrinth_controller import UnityLabyrinthController
from Controllers.unity_meta_controller import MetaController
import pickle
from datetime import datetime
from MDP.general_high_level_mdp import HLMDP
from utils.results_saver import Results
# from optimization_problems.high_level_reward_opt import solve_max_reward

import yaml

# %% Setup and create the environment

# Import the environment information
env_info_folder = os.path.abspath('../Environments')
env_info_file_name = 'unity_shuffle.yaml'
env_info_str = os.path.join(env_info_folder, env_info_file_name)
with open(env_info_str, 'rb') as f:
    env_info = yaml.safe_load(f)

env, side_channels = build_unity_labyrinth_env()

prob_threshold = 0.95 # Desired probability of reaching the final goal
training_iters = 5e4 # 5e4
num_rollouts = 10 # 100
n_steps_per_rollout = 500
max_timesteps_per_component = 2e5

controller_to_train_idx = 17

# %% Set the load directory (if loading pre-trained sub-systems) 
# or create a new directory in which to save results

load_folder_name = '2022-10-14_18-34-32_unity_labyrinth_shuffle_x_1_0_inherit_default_params'
save_learned_controllers = True

experiment_name = 'unity_labyrinth'

base_path = os.path.abspath(os.path.curdir)
string_ind = base_path.find('src')
assert(string_ind >= 0)
base_path = base_path[0:string_ind + 4]
base_path = os.path.join(base_path, 'data', 'saved_controllers')

load_dir = os.path.join(base_path, load_folder_name)


# %% Create the list of partially instantiated sub-systems
controller_list = []


for controller_dir in os.listdir(load_dir):
    controller_load_path = os.path.join(load_dir, controller_dir)
    if os.path.isdir(controller_load_path):
        controller = UnityLabyrinthController(0, env, load_dir=controller_load_path, verbose=True)
        controller_list.append(controller)
    
    # re-order the controllers by index
reordered_list = []
print('PRINT LENGTH : ', len(controller_list))
for i in range(len(controller_list)):
    for controller in controller_list:
        if controller.controller_ind == i:
            reordered_list.append(controller)    

controller_list = reordered_list

# %% Create or load object to store the results
# for i in range(len(controller_list)):
#     print('CONTROLLER INDEX : ', controller_list[i].controller_ind)

results = Results(load_dir=load_dir)
rseed = results.data['random_seed']

# %%

import torch
import random
torch.manual_seed(rseed)
random.seed(rseed)
np.random.seed(rseed)

print('Random seed: {}'.format(results.data['random_seed']))

# %%

# # Construct high-level MDP and solve for the max reach probability
S = np.arange(-1, env_info['N_S'] - 1)
A = np.arange(env_info['N_A'])
successor_map = {}
for key,val in env_info['successor_map'].items():
    newkey = tuple(int(x) for x in key.strip('[]').split(','))
    newval = val
    successor_map[newkey] = newval


# controller_list[1].controller_ind = 1
hlmdp = HLMDP(S, A, env_info["s_i"], env_info["s_goal"], env_info["s_fail"], controller_list, successor_map)
policy, reach_prob, feasible_flag = hlmdp.solve_max_reach_prob_policy()

side_channels['engine_config_channel'].set_configuration_parameters(time_scale=99.0)

num_rollouts = 10
n_steps_per_rollout = 500
render = True
meta_controller = MetaController(policy, hlmdp, side_channels)

controller_to_train = hlmdp.controller_list[controller_to_train_idx]
print('Training controller {}'.format(controller_to_train_idx))
controller_to_train.learn(side_channels['custom_side_channel'], 
                                total_timesteps=5e4)
print('Completed training controller {}'.format(controller_to_train_idx))
side_channels['engine_config_channel'].set_configuration_parameters(time_scale=1.0)
print('\n --Evaluating--')
controller_to_train.eval_performance(env, 
                                    side_channels['custom_side_channel'], 
                                    n_episodes=num_rollouts,
                                    n_steps=n_steps_per_rollout)
print('Controller {} achieved prob succes: {}'.format(controller_to_train_idx, 
                                                controller.get_success_prob()))

# flag = 1
# subController = controller_to_train_idx
# if flag == 1:
#     print('\n ---Demonstrating Sub Controller {}--- \n'.format(controller_to_train_idx))
#     side_channels['engine_config_channel'].set_configuration_parameters(time_scale=1.0)
#     controller = controller_list[subController]
#     # Evaluate initial performance of controllers (they haven't learned 
#     # anything yet so they will likely have no chance of success.)
#     controller_to_train.demonstrate_capabilities(env, 
#                                 side_channels['custom_side_channel'], 
#                                 n_episodes=2,
#                                 n_steps=n_steps_per_rollout)

# print('\n ---Demonstrating Meta Controller--- \n')
# meta_controller.demonstrate_capabilities(env, 
#                                         side_channels,
#                                         n_episodes=n_episodes, 
#                                         n_steps=n_steps_per_rollout, 
#                                         render=render)
# meta_controller.unsubscribe_meta_controller(side_channels)

