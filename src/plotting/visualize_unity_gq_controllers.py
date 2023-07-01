import os, sys
sys.path.append('..')

from environments.unity_env import build_unity_env
import numpy as np
from controllers.unity_controller import UnityController
from controllers.unity_meta_controller import MetaController
import pickle
from datetime import datetime
from MDP.general_high_level_mdp import HLMDP
from utils.results_saver import Results

import yaml

from utils.loaders import instantiate_controllers, load_env_info

from examples.gq_robotics.config.gq_12_subgoals_config import cfg

# Setup and create the environment
# # Import the environment information
env_info = load_env_info(cfg['hlmdp_file_name'])

env_settings = {
    'time_scale' : 99.0,
}

env, side_channels = build_unity_env()

# Set the load directory (if loading pre-trained sub-systems) 
# or create a new directory in which to save results

# load_folder_name = '2023-05-19_19-17-47_gq_mission_20_subgoalstrain_all_controllers_completed'
# load_folder_name = '2023-06-23_01-54-52_gq_mission_11_subgoals_linear3_angular1p5_left_route'
# load_folder_name = '2023-06-23_15-10-27_gq_mission_12_subgoals_composite_policy'
# load_folder_name = '2023-06-26_18-36-39_gq_mission_12_subgoals_composite_policy_left'
# load_folder_name = '2023-06-26_18-36-39_gq_mission_12_subgoals_composite_policy_right'
# load_folder_name = '2023-06-27_20-31-17_gq_mission_12_subgoals_composite_policy2'
# load_folder_name = '2023-06-28_13-53-29_gq_mission_12_subgoals_composite_policy_left_penalized_turns'
load_folder_name = '2023-06-30_00-02-29_gq_mission_12_subgoals_composite_policy_left_new_six_mixnmatch'

# experiment_name = 'gq_mission_20_subgoals'

base_path = os.path.abspath(os.path.join('..', 'examples/gq_robotics', 'data', 'saved_controllers'))

save_path = os.path.join(base_path, load_folder_name)

# # Create the list of partially instantiated sub-systems
controller_list = instantiate_controllers(env, env_settings=env_settings, load_dir=save_path)

# Load object to store the results
results = Results(load_dir=save_path)
rseed = results.data['random_seed']

import torch
import random
torch.manual_seed(rseed)
random.seed(rseed)
np.random.seed(rseed)

print('Random seed: {}'.format(results.data['random_seed']))

# Construct high-level MDP and solve for the max reach probability
S = np.arange(-1, env_info['N_S'] - 1)
A = np.arange(env_info['N_A'])
successor_map = {}
for key,val in env_info['successor_map'].items():
    newkey = tuple(int(x) for x in key.strip('[]').split(','))
    newval = val
    successor_map[newkey] = newval

hlmdp = HLMDP(S, A, env_info["s_i"], env_info["s_goal"], env_info["s_fail"], controller_list, successor_map)
policy, reach_prob, feasible_flag = hlmdp.solve_max_reach_prob_policy()

meta_policy = results.data['composition_policy'][np.max([key for key in results.data['composition_policy'].keys()])]

print(policy)
print(meta_policy)
# side_channels['engine_config_channel'].set_configuration_parameters(time_scale=1.0)

# Construct a meta-controller and emprirically evaluate it.
n_episodes = 5
n_steps_per_rollout = 7 * 400
render = True
meta_controller = MetaController(meta_policy, hlmdp, side_channels)
meta_controller.demonstrate_capabilities(env, 
                                        side_channels,
                                        n_episodes=n_episodes, 
                                        n_steps=n_steps_per_rollout, 
                                        render=render)
meta_controller.unsubscribe_meta_controller(side_channels)

# Construct a meta-controller with this policy and empirically evaluate its performance
num_rollouts = 100
side_channels['engine_config_channel'].set_configuration_parameters(time_scale=99.0)
meta_controller = MetaController(policy, hlmdp, side_channels)
meta_success_rate = meta_controller.eval_performance(env,
                                                    side_channels, 
                                                    n_episodes=num_rollouts, 
                                                    n_steps=n_steps_per_rollout)
meta_controller.unsubscribe_meta_controller(side_channels)

print(meta_success_rate)

env.close()