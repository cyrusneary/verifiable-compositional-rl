import os, sys
sys.path.append('../..')

from environments.unity_env import build_unity_env
import numpy as np
from controllers.unity_controller import UnityController
from controllers.unity_meta_controller import MetaController
import os, sys
from datetime import datetime
from MDP.general_high_level_mdp import HLMDP
from utils.results_saver import Results
import yaml
import pickle

import torch
import random

from utils.loaders import instantiate_controllers, load_env_info

from config.gq_2_subgoals_config import cfg

# Setup and create the environment
# Import the environment information (HLMDP Structure)
env_info_folder = os.path.abspath('../../environments')
env_info_str = os.path.join(env_info_folder, cfg['hlmdp_file_name'])
with open(env_info_str, 'rb') as f:
    env_info = yaml.safe_load(f)

env_settings = {
    'time_scale' : 99.0,
}

env, side_channels = build_unity_env()
side_channels['engine_config_channel'].set_configuration_parameters(
                                        time_scale=env_settings['time_scale'])

# Instantiate the RL controllers.
if cfg['controller_instantiation_method'] == 'new':
    # Instantiate an entirely new collection of RL-based controllers
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        cfg['log_settings']['base_save_dir'], 
        dt_string + '_' + cfg['experiment_name']
    )
    tensorboard_log = os.path.join(
        cfg['log_settings']['base_tensorboard_logdir'], 
        dt_string + '_' + cfg['experiment_name']
    )
    controller_list = instantiate_controllers(
        env, 
        env_settings=env_settings,
        num_controllers=env_info['N_A'],
        verbose=cfg['log_settings']['verbose'],
        tensorboard_log=tensorboard_log,
    )
elif cfg['controller_instantiation_method'] == 'pre_trained':
    # Instantiate a collection of RL-based controllers, initialized using some
    # pre-trained controller.
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        cfg['log_settings']['base_save_dir'], 
        dt_string + '_from_pre_trained_' + cfg['experiment_name']
    )
    load_path = os.path.join(
        cfg['log_settings']['base_save_dir'],
        cfg['load_folder_name']
    )
    tensorboard_log = os.path.join(
        cfg['log_settings']['base_tensorboard_logdir'], 
        dt_string + '_' + cfg['experiment_name']
    )
    controller_list = instantiate_controllers(
        env, 
        env_settings=env_settings,
        num_controllers=env_info['N_A'],
        pre_trained_load_dir=load_path,
        verbose=cfg['log_settings']['verbose'],
        tensorboard_log=tensorboard_log,
    )
elif cfg['controller_instantiation_method'] == 'load':
    # Load a collection of pre-existing controllers.
    save_path = os.path.join(
        cfg['log_settings']['base_save_dir'], 
        cfg['load_folder_name']
    )
    load_path = save_path
    tensorboard_log = os.path.join(
        cfg['log_settings']['base_tensorboard_logdir'], 
        cfg['load_folder_name']
    )
    controller_list = instantiate_controllers(
        env, 
        env_settings=env_settings,
        load_dir=load_path,
        verbose=cfg['log_settings']['verbose'],
        tensorboard_log=tensorboard_log,
    )
else:
    raise ValueError('Invalid controller_instantiation_method in config file.')

if not os.path.isdir(save_path):
    os.mkdir(save_path)

# Create or load object to store the results
if cfg['controller_instantiation_method'] == 'new' or \
    cfg['controller_instantiation_method'] == 'pre_trained':
    results = Results(controller_list, 
                        env_settings, 
                        cfg['icrl_parameters']['prob_threshold'], 
                        cfg['icrl_parameters']['training_iters'], 
                        cfg['icrl_parameters']['num_rollouts'] , 
                        random_seed=cfg['rseed'])
else:
    results = Results(load_dir=load_path)

torch.manual_seed(cfg['rseed'])
random.seed(cfg['rseed'])
np.random.seed(cfg['rseed'])

print('Random seed: {}'.format(results.data['random_seed']))

for controller_ind in range(len(controller_list)):
    controller = controller_list[controller_ind]
    # Evaluate initial performance of controllers (they haven't learned 
    # anything yet so they will likely have no chance of success.)
    controller.eval_performance(env, 
                                side_channels['custom_side_channel'], 
                                n_episodes=1,
                                n_steps=cfg['icrl_parameters']['n_steps_per_rollout'])
    print('Controller {} achieved prob succes: {}'.format(controller_ind, 
                                                controller.get_success_prob()))

    # Save learned controller
    controller_save_path = \
        os.path.join(save_path, 'controller_{}'.format(controller_ind))
    controller.save(controller_save_path)

results.update_training_steps(0)
results.update_controllers(controller_list)
results.save(save_path)

# Setup the high-level MDP object
S = np.arange(-1, env_info['N_S'] - 1)
A = np.arange(env_info['N_A'])
successor_map = {}
for key,val in env_info['successor_map'].items():
    newkey = tuple(int(x) for x in key.strip('[]').split(','))
    newval = val
    successor_map[newkey] = newval

hlmdp = HLMDP(S, A, env_info['s_i'], env_info['s_goal'], env_info['s_fail'], controller_list, successor_map)
policy, reach_prob, feasible_flag = hlmdp.solve_max_reach_prob_policy()

# Construct a meta-controller and emprirically evaluate it.
meta_controller = MetaController(policy, hlmdp, side_channels)
meta_success_rate = meta_controller.eval_performance(env,
                                                    side_channels,
                                                    n_episodes=cfg['icrl_parameters']['num_rollouts'] , 
                                                    n_steps=cfg['icrl_parameters']['meta_controller_n_steps_per_rollout'])
meta_controller.unsubscribe_meta_controller(side_channels)

# Save the results
results.update_composition_data(meta_success_rate, cfg['icrl_parameters']['num_rollouts'] , policy, reach_prob)
results.save(save_path)

# Main loop of iterative compositional reinforcement learning
while reach_prob < cfg['icrl_parameters']['prob_threshold']:

    # Solve the HLM biliniear program to obtain sub-task specifications.
    optimistic_policy, \
        required_reach_probs, \
            optimistic_reach_prob, \
                feasible_flag = \
                    hlmdp.solve_low_level_requirements_action(cfg['icrl_parameters']['prob_threshold'], 
                    max_timesteps_per_component=cfg['icrl_parameters']['max_timesteps_per_component'])

    if not feasible_flag:
        print(required_reach_probs)

    # Print the empirical sub-system estimates and the sub-system 
    # specifications to terminal
    for controller_ind in range(len(hlmdp.controller_list)):
        controller = hlmdp.controller_list[controller_ind]
        print('Sub-task: {}, \
                Achieved success prob: {}, Required success prob: {}'\
                    .format(controller_ind, 
                            controller.get_success_prob(), 
                            controller.data['required_success_prob']))

    # Decide which sub-system to train next.
    performance_gaps = []
    for controller_ind in range(len(hlmdp.controller_list)):
        controller = hlmdp.controller_list[controller_ind]
        performance_gaps.append(controller.data['required_success_prob'] - \
                                controller.get_success_prob())

    largest_gap_ind = np.argmax(performance_gaps)
    controller_to_train = hlmdp.controller_list[largest_gap_ind]

    # Train the sub-system and empirically evaluate its performance
    print('Training controller {}'.format(largest_gap_ind))
    controller_to_train.learn(side_channels['custom_side_channel'], 
                                total_timesteps=cfg['icrl_parameters']['training_iters'])
    print('Completed training controller {}'.format(largest_gap_ind))
    controller_to_train.eval_performance(env, 
                                        side_channels['custom_side_channel'], 
                                        n_episodes=cfg['icrl_parameters']['num_rollouts'] ,
                                        n_steps=cfg['icrl_parameters']['n_steps_per_rollout'])

    # Save learned controller
    controller_save_path = os.path.join(save_path, 
                                'controller_{}'.format(largest_gap_ind))
    if not os.path.isdir(controller_save_path):
        os.mkdir(controller_save_path)
    controller_to_train.save(controller_save_path)

    # Solve the HLM for the meta-policy maximizing reach probability
    policy, reach_prob, feasible_flag = hlmdp.solve_max_reach_prob_policy()

    # Construct a meta-controller with this policy and empirically evaluate its performance
    meta_controller = MetaController(policy, hlmdp, side_channels)
    meta_success_rate = meta_controller.eval_performance(env,
                                                        side_channels, 
                                                        n_episodes=cfg['icrl_parameters']['num_rollouts'] , 
                                                        n_steps=cfg['icrl_parameters']['meta_controller_n_steps_per_rollout'])
    meta_controller.unsubscribe_meta_controller(side_channels)

    # Save results
    results.update_training_steps(cfg['icrl_parameters']['training_iters'])
    results.update_controllers(hlmdp.controller_list)
    results.update_composition_data(meta_success_rate, cfg['icrl_parameters']['num_rollouts'], policy, reach_prob)
    results.save(save_path)

    print('Predicted success prob: {}, Empirical success prob: {}'.format(reach_prob, meta_success_rate))

# Once the loop has been completed, construct a meta-controller and visualize its performance

meta_controller = MetaController(policy, hlmdp, side_channels)
print('evaluating performance of meta controller')
meta_success_rate = meta_controller.eval_performance(env,
                                                    side_channels,
                                                    n_episodes=cfg['icrl_parameters']['num_rollouts'], 
                                                    n_steps=cfg['icrl_parameters']['meta_controller_n_steps_per_rollout'])
meta_controller.unsubscribe_meta_controller(side_channels)
print('Predicted success prob: {}, \
    empirically measured success prob: {}'.format(reach_prob, meta_success_rate))

n_episodes = 5
render = True
meta_controller = MetaController(policy, hlmdp, side_channels)
meta_controller.demonstrate_capabilities(env, 
                                        side_channels,
                                        n_episodes=n_episodes, 
                                        n_steps=cfg['icrl_parameters']['meta_controller_n_steps_per_rollout'], 
                                        render=render)
meta_controller.unsubscribe_meta_controller(side_channels)

env.close()