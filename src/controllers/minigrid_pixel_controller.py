import numpy as np
from numpy.core.numeric import roll
from stable_baselines3 import PPO
from environments.minigrid_pixel_labyrinth import PixelMaze
import os, sys
from datetime import datetime
import pickle

from .minigrid_controller import MiniGridController

class MiniGridPixelController(MiniGridController):
    """
    Class representing PPO-based controllers that learn to accomplish goal-oriented sub-tasks within
    the minigrid gym environment.
    """

    def __init__(self, controller_ind, init_states=None, final_states=None, env_settings=None, max_training_steps=1e6, load_dir=None, verbose=False):
        
        super().__init__(controller_ind, init_states, final_states, env_settings, max_training_steps, load_dir, verbose)

        self.task_complete_obs = []
        for final_state in self.training_env.goal_states:
            self.training_env.agent_pos = (final_state[0], final_state[1])
            self.training_env.agent_dir = final_state[2]
            self.task_complete_obs.append(self.training_env.gen_obs())

    def _set_training_env(self, env_settings):
        self.training_env = PixelMaze(**env_settings)
        self.training_env.agent_start_states = self.init_states
        self.training_env.goal_states = self.final_states

    def _init_learning_alg(self, verbose=False):
        self.model = PPO("CnnPolicy", 
                            self.training_env, 
                            verbose=verbose,
                            n_steps=512,
                            batch_size=64,
                            gae_lambda=0.95,
                            gamma=0.95,
                            n_epochs=10,
                            ent_coef=0.0,
                            learning_rate=2.5e-4,
                            clip_range=0.2)

    def is_task_complete(self, state):
        """
        Return true if the current observation indicates the agent has already reached its goal.
        """
        current_state = (state[0], state[1], state[2])
        if current_state in self.final_states:
            return True
        else:
            return False

    # def is_task_complete(self, obs):
    #     for task_complete_obs in self.task_complete_obs:
    #         if (np.array(obs) == np.array(task_complete_obs)).all():
    #             return True
    #     else:
    #         return False

    def demonstrate_capabilities(self, n_episodes=5, n_steps=100, render=True):
        """
        Demonstrate the capabilities of the learned controller in the environment used to train it.

        Inputs
        ------
        n_episodes : int
            Number of episodes to rollout for evaluation.
        n_steps : int
            Length of each episode.
        render (optional) : bool
            Whether or not to render the environment at every timestep.
        """
        for episode_ind in range(n_episodes):
            obs = self.training_env.reset()
            for step in range(n_steps):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.training_env.step(action)
                if render:
                    self.training_env.render(highlight=True)
                if done:
                    break

        obs = self.training_env.reset()