
cfg = {
    'hlmdp_file_name': 'unity_labyrinth.yaml',
    'icrl_parameters': {
        'prob_threshold': 0.95,
        'training_iters': 5e4,
        'num_rollouts': 100,
        'n_steps_per_rollout': 100,
        'meta_controller_n_steps_per_rollout': 5 * 100,
        'max_timesteps_per_component': 2e5
    }
}