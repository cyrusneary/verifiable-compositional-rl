
# Run the labyrinth navigation experiment.
import os, sys
sys.path.append('..')

from stable_baselines3.common.callbacks import CheckpointCallback

from environments.unity_env import build_unity_env
import numpy as np
from controllers.unity_controller import UnityController
import os, sys
from datetime import datetime
from utils.results_saver import Results
import yaml

import torch
import random

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
side_channels['engine_config_channel'].set_configuration_parameters(
                                        time_scale=env_settings['time_scale'])

training_iters = 1e6
prob_threshold = cfg['icrl_parameters']['prob_threshold'] # Desired probability of reaching the final goal
num_rollouts = cfg['icrl_parameters']['num_rollouts'] 
n_steps_per_rollout = cfg['icrl_parameters']['n_steps_per_rollout']
meta_controller_n_steps_per_rollout = cfg['icrl_parameters']['meta_controller_n_steps_per_rollout']
max_timesteps_per_component = cfg['icrl_parameters']['max_timesteps_per_component']

# Set the load directory (if loading pre-trained sub-systems) 
# or create a new directory in which to save results
load_folder_name = ''
save_learned_controllers = True

experiment_name = env_info_file_name.split('.yaml')[0] + 'train_all_controllers'

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

if save_learned_controllers and not os.path.isdir(save_path):
    os.mkdir(save_path)

# Create the list of partially instantiated sub-systems
controller_list = []

base_tensorboard_folder = './tensorboard/'

if load_folder_name == '':
    for i in range(env_info['N_A']):
        tensorboard_folder = base_tensorboard_folder + 'controller_{}'.format(i)
        controller_list.append(
            UnityController(
                i, 
                env, 
                env_settings=env_settings, 
                verbose=True,
                tensorboard_log=tensorboard_folder,
            )
        )
else:
    for controller_dir in os.listdir(load_dir):
        controller_load_path = os.path.join(load_dir, controller_dir)
        if os.path.isdir(controller_load_path):
            controller = UnityController(
                0, 
                env, 
                load_dir=controller_load_path, 
                verbose=True
            )
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

torch.manual_seed(rseed)
random.seed(rseed)
np.random.seed(rseed)

print('Random seed: {}'.format(results.data['random_seed']))

for controller_ind in range(len(controller_list)):
    controller = controller_list[controller_ind]
    # Evaluate initial performance of controllers (they haven't learned 
    # anything yet so they will likely have no chance of success.)
    controller.eval_performance(env, 
                                side_channels['custom_side_channel'], 
                                n_episodes=1,
                                n_steps=n_steps_per_rollout)
    print('Controller {} achieved prob succes: {}'.format(controller_ind, 
                                                controller.get_success_prob()))

    # Save learned controller
    if save_learned_controllers:
        controller_save_path = \
            os.path.join(save_path, 'controller_{}'.format(controller_ind))
        controller.save(controller_save_path)

results.update_training_steps(0)
results.update_controllers(controller_list)
results.save(save_path)

for controller_ind in range(len(controller_list)):
    controller = controller_list[controller_ind]

    # Save learned controller
    controller_save_path = \
        os.path.join(save_path, 'controller_{}'.format(controller_ind))
    
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1e4,
        save_path=controller_save_path,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Train the sub-system and empirically evaluate its performance
    print('Training controller {}'.format(controller_ind))
    controller.learn(
        side_channels['custom_side_channel'], 
        total_timesteps=training_iters,
        callback=checkpoint_callback
    )
    print('Completed training controller {}'.format(controller_ind))
    controller.eval_performance(env, 
                                side_channels['custom_side_channel'], 
                                n_episodes=num_rollouts,
                                n_steps=n_steps_per_rollout)

    controller.save(controller_save_path)

    results.update_training_steps(training_iters)
    results.update_controllers(controller_list)
    results.save(save_path)

env.close()