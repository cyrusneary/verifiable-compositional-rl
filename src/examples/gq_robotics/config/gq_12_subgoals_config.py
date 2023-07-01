import os

base_path = os.path.abspath('/home/cyrusneary/Documents/research/code') # local machine

cfg = {
    'experiment_name' : 'gq_mission_12_subgoals',
    'hlmdp_file_name': 'gq_mission_12_subgoals.yaml',
    'rseed' : 42,
    'icrl_parameters': {
        'prob_threshold': 0.95,
        'training_iters': 1e6,
        'num_rollouts': 100,
        'n_steps_per_rollout': 600,
        'meta_controller_n_steps_per_rollout': 7 * 600,
        'max_timesteps_per_component': 1e7
    },
    'controller_instantiation_method' : 'load', # ['new', 'load', 'pre_trained']
    'load_folder_name' : '2023-06-30_00-02-29_gq_mission_12_subgoals_composite_policy_left_new_six_mixnmatch', #'2023-06-28_13-53-29_gq_mission_12_subgoals_composite_policy_penalized_turns', #'2023-06-27_20-31-17_gq_mission_12_subgoals_composite_policy2', #'2023-06-26_18-36-39_gq_mission_12_subgoals_composite_policy_right',
    'log_settings' : {
        'verbose' : True,
        'base_save_dir' : 
            os.path.abspath(
                os.path.join(
                    base_path,
                    'verifiable-compositional-rl/src/examples/gq_robotics/data/saved_controllers/'
                )
            ),
        'base_tensorboard_logdir' : 
            os.path.abspath(
                os.path.join(
                    base_path, 
                    'verifiable-compositional-rl/src/examples/gq_robotics/tensorboard'
                )
            ),
    }
    
}