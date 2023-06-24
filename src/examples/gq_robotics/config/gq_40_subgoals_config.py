import os

base_path = os.path.abspath('/home/cyrusneary/Documents/research/code') # local machine

cfg = {
    'experiment_name' : 'gq_mission_40_subgoals',
    'hlmdp_file_name': 'gq_mission_40_subgoals.yaml',
    'rseed' : 42,
    'icrl_parameters': {
        'prob_threshold': 0.9,
        'training_iters': 5e5,
        'num_rollouts': 100,
        'n_steps_per_rollout': 400,
        'meta_controller_n_steps_per_rollout': 7 * 400,
        'max_timesteps_per_component': 1e7
    },
    'controller_instantiation_method' : 'load', # ['new', 'load', 'pre_trained']
    'load_folder_name' : '2023-06-22_17-33-34_gq_mission_40_subgoalstrain_all_controllers_copy', #'2023-05-20_16-32-24_pretrain_warthog_controller', #'2023-05-19_19-17-47_gq_mission_20_subgoalstrain_all_controllers_completed',
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