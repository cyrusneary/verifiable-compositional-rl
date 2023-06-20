
import os, sys
sys.path.append('../..')

from datetime import datetime

from environments.unity_env import build_unity_env
import numpy as np  
import pickle
import os, sys
from datetime import datetime
from utils.results_saver import Results
import yaml

from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Give the experiment a name
experiment_name = 'gq_simple_nav_task'
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save directory for the learned model
save_dir = os.path.abspath(os.path.join('results', 'saved_models', current_time + '_' + experiment_name))
if not os.path.isdir(save_dir): 
    os.makedirs(save_dir)

env_settings = {
    'time_scale' : 99.0,
}

env, side_channels = build_unity_env()
obs = env.reset()
side_channels['engine_config_channel'].set_configuration_parameters(
                                        time_scale=env_settings['time_scale'])

# Setup the reinforcement learning algorithm
model = PPO("MlpPolicy", 
                env, 
                verbose=True,
                n_steps=512,
                batch_size=64,
                gae_lambda=0.95,
                gamma=0.99,
                n_epochs=10,
                ent_coef=0.0,
                learning_rate=2.5e-4,
                clip_range=0.2,
                tensorboard_log="./tensorboard/")

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1e4,
  save_path=save_dir,
  name_prefix="rl_model",
  save_replay_buffer=False,
  save_vecnormalize=False,
)

# Train the model
model.learn(total_timesteps=2e7, callback=checkpoint_callback)

# Save the final version of the model.
model_file = os.path.join(save_dir, 'final_model')
model.save(model_file)

# Demonstrate the learned controller
n_episodes = 5
n_steps = 300

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