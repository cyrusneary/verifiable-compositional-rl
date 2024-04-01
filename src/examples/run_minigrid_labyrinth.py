
# Run the labyrinth navigation experiment.

# %%
import os, sys
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

# %% Setup and create the environment
env_settings = {
    'agent_start_states' : [(22,22,0)],
    'slip_p' : 0.1,
}

env = Maze(**env_settings)

prob_threshold = 0.95 # Desired probability of reaching the final goal
training_iters = 5e4
num_rollouts = 300
max_timesteps_per_component = 5e5

n_steps_per_rollout = 100
meta_controller_n_steps_per_rollout = 200

# %% Set the load directory (if loading pre-trained sub-systems) or create a new directory in which to save results

load_folder_name = ''
save_learned_controllers = True

experiment_name = 'minigrid_labyrinth'

base_path = os.path.abspath(os.path.curdir)
string_ind = base_path.find('src')
assert(string_ind >= 0)
base_path = base_path[0:string_ind + 4]
base_path = os.path.join(base_path, 'data', 'saved_controllers')

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

# %% Create the list of partially instantiated sub-systems
controller_list = []

if load_folder_name == '':

    '''
    # Top left L-shape room #1
    initial_states = [(2,2,0)]
    final_states = [(12,2,0)]   
    controller_list.append(MiniGridController(0, initial_states, final_states, env_settings))

    initial_states = [(12,2,0)]
    final_states = [(2,2,0)]  
    controller_list.append(MiniGridController(1, initial_states, final_states, env_settings))

    # Top right L-shape room #2
    initial_states = [(12,2,0)]
    final_states = [(22,2,0)]    
    controller_list.append(MiniGridController(2, initial_states, final_states, env_settings))
    
    initial_states = [(22,2,0)]
    final_states = [(12,2, 0)]   
    controller_list.append(MiniGridController(3, initial_states, final_states, env_settings))

    # middle top room #3
    initial_states = [(12,2,0)]
    final_states = [(12,6,0)]   
    controller_list.append(MiniGridController(4, initial_states, final_states, env_settings))

    initial_states = [(12,6,0)] 
    final_states = [(12,2, 0)]   
    controller_list.append(MiniGridController(5, initial_states, final_states, env_settings))

    # Left top middle room #4 
    initial_states = [(2,6,0)]
    final_states = [(12,6,0)]
    controller_list.append(MiniGridController(6, initial_states, final_states, env_settings))

    initial_states = [(12,6,0)]
    final_states = [(2,6,0)]
    controller_list.append(MiniGridController(7, initial_states, final_states, env_settings))
    # Left middle room #5 
    initial_states = [(2,6,0)]
    final_states = [(2,12,0)]
    controller_list.append(MiniGridController(8, initial_states, final_states, env_settings))
    
    initial_states = [(2,12,0)]
    final_states = [(2,6,0)]
    controller_list.append(MiniGridController(9, initial_states, final_states, env_settings))

    # Middle middle room #6 
    initial_states = [(12,6,0)]
    final_states = [(12,12,0)]
    controller_list.append(MiniGridController(10, initial_states, final_states, env_settings))

    initial_states = [(12,12,0)]
    final_states = [(12,6,0)]   
    controller_list.append(MiniGridController(11, initial_states, final_states, env_settings))

    # Middle middle room #7 
    initial_states = [(2,12,0)]
    final_states = [(12,12,0)]   
    controller_list.append(MiniGridController(12, initial_states, final_states, env_settings))
    initial_states = [(12,12,0)]
    final_states = [(2,12,0)]  
    controller_list.append(MiniGridController(13, initial_states, final_states, env_settings))
    
    # Right middle room #8
    initial_states = [(12,12,0)]
    final_states = [(22,12,0)]
    controller_list.append(MiniGridController(14, initial_states, final_states, env_settings))

    initial_states = [(22,12,0)]
    final_states = [(12,12,0)]
    controller_list.append(MiniGridController(15, initial_states, final_states, env_settings))

    # Bottom left L-shape room #9
    initial_states = [(2,12,0)]
    final_states = [(2,22,0)] 
    controller_list.append(MiniGridController(16, initial_states, final_states, env_settings))
    
    initial_states = [(2,22,0)]
    final_states = [(2,12,0)] 
    controller_list.append(MiniGridController(17, initial_states, final_states, env_settings))

    # Bottom middle room #10 
    initial_states = [(12,12,0)]
    final_states = [(12,22,0)] 
    controller_list.append(MiniGridController(18, initial_states, final_states, env_settings))
    
    initial_states = [(12,22,0)]
    final_states = [(12,12,0)] 
    controller_list.append(MiniGridController(19, initial_states, final_states, env_settings))

    # Bottom right L-shape room #11 
    initial_states = [(22,12,0)]
    final_states = [(22,22,0)] 
    controller_list.append(MiniGridController(20, initial_states, final_states, env_settings))
    
    initial_states = [(22,22,0)]
    final_states = [(22,12,0)] 
    controller_list.append(MiniGridController(21, initial_states, final_states, env_settings))
    #
    initial_states = [(12,22,0)]
    final_states = [(22,22,0)] 
    controller_list.append(MiniGridController(22, initial_states, final_states, env_settings))

    '''
    initial_states = [(22,22,0)]
    final_states = [(12,22,0)] 
    controller_list.append(MiniGridController(23, initial_states, final_states, env_settings))
    '''
    #
    initial_states = [(2,22,0)]
    final_states = [(12,22,0)] 
    controller_list.append(MiniGridController(24, initial_states, final_states, env_settings))

    '''
    initial_states = [(12,22,0)]
    final_states = [(2,22,0)] 
    controller_list.append(MiniGridController(25, initial_states, final_states, env_settings))
    '''

    #
    initial_states = [(2,6,0)]
    final_states = [(2,2,0)] 
    controller_list.append(MiniGridController(26, initial_states, final_states, env_settings))

    initial_states = [(2,2,0)]
    final_states = [(2,6,0)] 
    controller_list.append(MiniGridController(27, initial_states, final_states, env_settings))
    #
    initial_states = [(22,2,0)]
    final_states = [(22,12,0)] 
    controller_list.append(MiniGridController(28, initial_states, final_states, env_settings))

    initial_states = [(22,12,0)]
    final_states = [(22,2,0)] 
    controller_list.append(MiniGridController(29, initial_states, final_states, env_settings))
    '''
else:
    for controller_dir in os.listdir(load_dir):
        controller_load_path = os.path.join(load_dir, controller_dir)
        if os.path.isdir(controller_load_path):
            controller = MiniGridController(0, load_dir=controller_load_path)
            controller_list.append(controller)

    # re-order the controllers by index
    reordered_list = []
    for i in range(len(controller_list)):
        for controller in controller_list:
            if controller.controller_ind == i:
                reordered_list.append(controller)
    controller_list = reordered_list

# %% Create or load object to store the results
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

# %%

import torch
import random
torch.manual_seed(rseed)
random.seed(rseed)
np.random.seed(rseed)

print('Random seed: {}'.format(results.data['random_seed']))

for controller_ind in range(len(controller_list)):
    controller = controller_list[controller_ind]
    # Evaluate initial performance of controllers (they haven't learned anything yet so they will likely have no chance of success.)
    controller.eval_performance(n_episodes=num_rollouts, n_steps=n_steps_per_rollout)
    print('Controller {} achieved prob succes: {}'.format(controller_ind, controller.get_success_prob()))

    # Save learned controller
    if save_learned_controllers:
        controller_save_path = os.path.join(save_path, 'controller_{}'.format(controller_ind))
        controller.save(controller_save_path)

results.update_training_steps(0)
results.update_controllers(controller_list)
results.save(save_path)
# %%

# Construct high-level MDP and solve for the max reach probability
hlmdp = HLMDP([(22,22,0)], env.goal_states, controller_list)

#liuc
#for s in hlmdp.S:
#    print ('s ', s, ' act ', hlmdp.avail_actions[s])


policy, reach_prob, feasible_flag = hlmdp.solve_max_reach_prob_policy()

# Construct a meta-controller and emprirically evaluate it.
meta_controller = MetaController(policy, hlmdp.controller_list, hlmdp.state_list)
meta_success_rate = meta_controller.eval_performance(env, n_episodes=num_rollouts, n_steps=meta_controller_n_steps_per_rollout)

# Save the results
results.update_composition_data(meta_success_rate, num_rollouts, policy, reach_prob)
results.save(save_path)

# %% Main loop of iterative compositional reinforcement learning

total_timesteps = training_iters

while reach_prob < prob_threshold:

    # Solve the HLM biliniear program to automatically optain sub-task specifications.
    optimistic_policy, required_reach_probs, optimistic_reach_prob, feasible_flag = hlmdp.solve_low_level_requirements_action(prob_threshold, max_timesteps_per_component=max_timesteps_per_component)

    print(required_reach_probs)


    if not feasible_flag:
        print(required_reach_probs)

    # Print the empirical sub-system estimates and the sub-system specifications to terminal
    for controller_ind in range(len(hlmdp.controller_list)):
        controller = hlmdp.controller_list[controller_ind]
        print('Init state: {}, Action: {}, End state: {}, Achieved success prob: {}, Required success prob: {}'.format(controller.get_init_states(), controller_ind, controller.get_final_states(), controller.get_success_prob(), controller.data['required_success_prob']))

    # Decide which sub-system to train next.
    performance_gaps = []
    for controller_ind in range(len(hlmdp.controller_list)):
        controller = hlmdp.controller_list[controller_ind]
        performance_gaps.append(controller.data['required_success_prob'] - controller.get_success_prob())

    largest_gap_ind = np.argmax(performance_gaps)
    controller_to_train = hlmdp.controller_list[largest_gap_ind]

    # Train the sub-system and empirically evaluate its performance
    print('Training controller {}'.format(largest_gap_ind))
    controller_to_train.learn(total_timesteps=total_timesteps)
    print('Completed training controller {}'.format(largest_gap_ind))
    controller_to_train.eval_performance(n_episodes=num_rollouts, n_steps=n_steps_per_rollout)

    # Save learned controller
    if save_learned_controllers:
        controller_save_path = os.path.join(save_path, 'controller_{}'.format(largest_gap_ind))
        if not os.path.isdir(controller_save_path):
            os.mkdir(controller_save_path)
        controller_to_train.save(controller_save_path)

    # Solve the HLM for the meta-policy maximizing reach probability
    policy, reach_prob, feasible_flag = hlmdp.solve_max_reach_prob_policy()

    # Construct a meta-controller with this policy and empirically evaluate its performance
    meta_controller = MetaController(policy, hlmdp.controller_list, hlmdp.state_list)
    meta_success_rate = meta_controller.eval_performance(env, n_episodes=num_rollouts, n_steps=meta_controller_n_steps_per_rollout)

    # Save results
    results.update_training_steps(total_timesteps)
    results.update_controllers(hlmdp.controller_list)
    results.update_composition_data(meta_success_rate, num_rollouts, policy, reach_prob)
    results.save(save_path)

# %% Once the loop has been completed, construct a meta-controller and visualize its performance

meta_controller = MetaController(policy, hlmdp.controller_list, hlmdp.state_list)
print('evaluating performance of meta controller')
meta_success_rate = meta_controller.eval_performance(env, n_episodes=num_rollouts, n_steps=meta_controller_n_steps_per_rollout)
print('Predicted success prob: {}, empirically measured success prob: {}'.format(reach_prob, meta_success_rate))

n_episodes = 5
n_steps = 200
render = True
meta_controller.demonstrate_capabilities(env, n_episodes=n_episodes, n_steps=n_steps, render=render)
