import os, sys
sys.path.append('..')

from environments.unity_env import build_unity_env
import numpy as np
from controllers.unity_labyrinth_controller import UnityLabyrinthController
from controllers.unity_meta_controller import MetaController
import pickle
from datetime import datetime
from MDP.general_high_level_mdp import HLMDP
from utils.results_saver import Results

import yaml

from examples.gq_robotics.config.gq_20_subgoals_config import cfg

# Setup and create the environment

# Import the environment information
env_info_file_name = cfg['hlmdp_file_name']
env_info_folder = os.path.abspath('../environments')
env_info_str = os.path.join(env_info_folder, env_info_file_name)
with open(env_info_str, 'rb') as f:
    env_info = yaml.safe_load(f)

env_settings = {
    'time_scale' : 99.0,
}

env, side_channels = build_unity_env()

prob_threshold = 0.95 # Desired probability of reaching the final goal
training_iters = 5e4 # 5e4
num_rollouts = 100 # 100
n_steps_per_rollout = 600
max_timesteps_per_component = 2e5

# Set the load directory (if loading pre-trained sub-systems) 
# or create a new directory in which to save results

load_folder_name = '2023-05-19_19-17-47_gq_mission_20_subgoalstrain_all_controllers_completed'
save_learned_controllers = True

experiment_name = 'gq_mission_20_subgoals'

base_path = os.path.abspath(os.path.curdir)
string_ind = base_path.find('src')
assert(string_ind >= 0)
base_path = base_path[0:string_ind + 4]
base_path = os.path.join(base_path, 'examples/gq_robotics', 'data', 'saved_controllers')

load_dir = os.path.join(base_path, load_folder_name)

if load_folder_name == '':
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    rseed = int(now.time().strftime('%H%M%S'))
    save_path = os.path.join(base_path, dt_string + '_' + experiment_name)
else:
    save_path = os.path.join(base_path, load_folder_name)

# Create the list of partially instantiated sub-systems
controller_list = []

if load_folder_name == '':
    for i in range(env_info['N_A']):
        controller_list.append(UnityLabyrinthController(i, env, env_settings=env_settings))
else:
    for controller_dir in os.listdir(load_dir):
        controller_load_path = os.path.join(load_dir, controller_dir)
        if os.path.isdir(controller_load_path):
            controller = UnityLabyrinthController(0, env, load_dir=controller_load_path)
            controller_list.append(controller)

    # re-order the controllers by index
    reordered_list = []
    for i in range(len(controller_list)):
        for controller in controller_list:
            if controller.controller_ind == i:
                reordered_list.append(controller)
    controller_list = reordered_list

# Create or load object to store the results
if load_folder_name == '':
    results = Results(controller_list, 
                        env_settings, 
                        prob_threshold, 
                        training_iters, 
                        num_rollouts, 
                        random_seed=rseed)
else:
    results = Results(load_dir=load_dir)
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

# side_channels['engine_config_channel'].set_configuration_parameters(time_scale=1.0)

# Construct a meta-controller and emprirically evaluate it.
n_episodes = 5
n_steps_per_rollout = 600
render = True
meta_controller = MetaController(policy, hlmdp, side_channels)
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