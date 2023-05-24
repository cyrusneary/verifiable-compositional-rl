
# Run the labyrinth navigation experiment.
import os, sys
sys.path.append('../..')
from stable_baselines3.common.callbacks import CheckpointCallback

from environments.unity_env import build_unity_env
import numpy as np
from controllers.unity_controller import UnityController
import os, sys
from datetime import datetime

import torch
import random

from config.pre_train_warthog_controller import cfg

# Setup and create the environment
env_settings = {
    'time_scale' : 99.0,
}

env, side_channels = build_unity_env()
side_channels['engine_config_channel'].set_configuration_parameters(
                                        time_scale=env_settings['time_scale'])

# Set the load directory (if loading pre-trained sub-systems) 
# or create a new directory in which to save results
load_folder_name = ''

base_path = os.path.abspath(os.path.curdir)
string_ind = base_path.find('src')
assert(string_ind >= 0)
base_path = base_path[0:string_ind + 4]
base_path = cfg['log_settings']['base_save_dir']

if cfg['load_folder_name'] == '':
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(base_path, dt_string + '_' + cfg['experiment_name'])
    tensorboard_log = os.path.join(
        cfg['log_settings']['base_tensorboard_logdir'],
        dt_string + '_' + cfg['experiment_name']
    )
else:
    save_path = os.path.join(base_path, load_folder_name)
    tensorboard_log = os.path.join(
        cfg['log_settings']['base_tensorboard_logdir'],
        load_folder_name
    )

if not os.path.isdir(save_path):
    os.mkdir(save_path)

# Create the list of partially instantiated sub-systems


if load_folder_name == '':
    controller = UnityController(
        0,
        env,
        env_settings=env_settings,
        verbose=cfg['log_settings']['verbose'],
        tensorboard_log=tensorboard_log,
    )
else:
    controller_dir = 'pretrained_controller'
    controller_load_path = os.path.join(save_path, controller_dir)
    controller = UnityController(
                0, 
                env, 
                load_dir=controller_load_path, 
                verbose=cfg['log_settings']['verbose'],
                tensorboard_log=tensorboard_log,
            )

torch.manual_seed(cfg['rseed'])
random.seed(cfg['rseed'])
np.random.seed(cfg['rseed'])

print('Random seed: {}'.format(cfg['rseed']))

# Save learned controller
controller_save_path = \
    os.path.join(save_path, 'pretrained_controller')

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=1e4,
    save_path=controller_save_path,
    name_prefix="checkpoint",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

# Train the sub-system and empirically evaluate its performance
print('Training controller')
controller.learn(
    side_channels['custom_side_channel'], 
    total_timesteps=cfg['training_iters'],
    callback=checkpoint_callback
)
print('Completed training controller')

controller.save(controller_save_path)

env.close()