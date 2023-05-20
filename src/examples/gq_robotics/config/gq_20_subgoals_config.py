
cfg = {
    'hlmdp_file_name': 'gq_mission_20_subgoals.yaml',
    'icrl_parameters': {
        'prob_threshold': 0.9,
        'training_iters': 1e5,
        'num_rollouts': 100,
        'n_steps_per_rollout': 400,
        'meta_controller_n_steps_per_rollout': 7 * 400,
        'max_timesteps_per_component': 1e7
    }
}