import os

base_path = os.path.abspath('/home/cyrusneary/Documents/research/code') # local machine

cfg = {
    'experiment_name' : 'pre_train_warthog_controller',
    'training_iters' : 1e7,
    'rseed' : 42,
    'load_folder_name' : '',
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