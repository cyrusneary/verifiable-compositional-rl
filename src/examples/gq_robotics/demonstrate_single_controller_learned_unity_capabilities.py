
import os, sys
sys.path.append('../..')

from environments.unity_env import build_unity_env
import numpy as np
import pickle
import os, sys
from datetime import datetime
from utils.results_saver import Results
import yaml

from tqdm import tqdm

from stable_baselines3 import PPO

experiment_name = '2023-05-20_16-32-24_pretrain_warthog_controller'
num_steps = 2.8e5

env_settings = {
    'time_scale' : 1.0,
}

env, side_channels = build_unity_env()
obs = env.reset()

# Load the previously trained model
# save_dir = os.path.join('results', 'saved_models')
# model_file = os.path.abspath(os.path.join(save_dir, experiment_name, 'rl_model_' + str(int(num_steps)) + '_steps.zip'))
save_dir = os.path.join('data', 'saved_controllers')
model_file = os.path.abspath(os.path.join(save_dir, experiment_name, 'pretrained_controller', 'checkpoint_' + str(int(num_steps)) + '_steps.zip'))

print(model_file)

model = PPO.load(model_file)

# Demonstrate the learned controller
n_episodes = 15
n_steps = 400

side_channels['engine_config_channel'].set_configuration_parameters(
                                        time_scale=1.0)

for episode_ind in range(n_episodes):
    obs = env.reset()
    for step in range(n_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            break

env.close()