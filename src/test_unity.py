from mlagents_envs.environment import UnityEnvironment, ActionTuple
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel import engine_configuration_channel
from stable_baselines3 import PPO
import numpy as np
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel
from utils.unity_utils import CustomSideChannel

from Environments.unity_labyrinth import build_unity_labyrinth_env

# engine_config_channel = EngineConfigurationChannel()
# custom_side_channel = CustomSideChannel()
# unity_env = UnityEnvironment(side_channels=[engine_config_channel, 
#                                                 custom_side_channel])

# env = UnityToGymWrapper(unity_env)

env_settings = {
    'time_scale' : 20.0,
}

env, side_channels = build_unity_labyrinth_env()
side_channels['engine_config_channel'].set_configuration_parameters(
                                        time_scale=env_settings['time_scale'])
side_channels['custom_side_channel'].send_string('2,2')

env.reset()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000)
model.save("ppo_reach_goal")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_reach_goal")

obs = env.reset()
side_channels['engine_config_channel'].set_configuration_parameters(time_scale = 1.0)
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    # env.render()

# for i in range(100):
#     env.step(env.action_space.sample())
# env.close()

# print('bla')
# env.reset()

# behavior_specs = env.behavior_specs

# behavior_names = list(env.behavior_specs.keys())
# behavior_name = behavior_names[0] # Only expecting one behavior

# for i in range(100):
#     random_action = np.array(-1*np.random.random((1, 2)), dtype=np.float32)
#     action_tuple = ActionTuple()
#     action_tuple.add_continuous(random_action)
#     env.set_actions(behavior_name=behavior_name, action=action_tuple)
#     env.step()