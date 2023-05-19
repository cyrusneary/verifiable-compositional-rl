import numpy as np
from numpy.core.numeric import roll
from stable_baselines3 import PPO
import os, sys
from datetime import datetime
import pickle
from utils.observers import ObserverIncrementTaskSuccessCount

class UnityLabyrinthController(object):
    """
    Class representing PPO-based controllers that learn to accomplish 
    goal-oriented sub-tasks within the unity labyrinth gym environment.
    """

    def __init__(self, 
                controller_ind, 
                env,
                env_settings=None, 
                max_training_steps=1e6, 
                load_dir=None, 
                verbose=False):
        
        self.controller_ind = controller_ind
        self.env_settings = env_settings
        self.verbose = verbose
        self.max_training_steps = max_training_steps

        self.data = {
            'total_training_steps' : 0,
            'performance_estimates' : {},
            'required_success_prob' : 0,
        }

        if load_dir is None:
            assert env_settings is not None
            self._init_learning_alg(env, verbose=self.verbose)
        else:
            self.load(env, load_dir)

    def learn(self, side_channel, total_timesteps=5e4):
        """
        Train the sub-system for a specified number of timesteps.

        Inputs
        ------
        side_channel : CustomSideChannel object
            An observable object that receives methods from the unity
            environment and notifies all subscribed observers.
        total_timesteps : int
            Total number of timesteps to train the sub-system for.
        """
        # Set the environment to the appropriate subtask
        sub_task_string = '{},{}'.format(self.controller_ind, self.controller_ind)
        side_channel.send_string(sub_task_string)

        self.model.learn(total_timesteps=total_timesteps)
        self.data['total_training_steps'] = self.data['total_training_steps'] \
                                                        + total_timesteps

    def predict(self, obs, deterministic=True):
        """
        Get the sub-system's action, given the current environment state

        Inputs
        ------
        obs : tuple
            Tuple representing the current environment observation (state).
        deterministic (optional) : bool
            Flag indicating whether or not to return a deterministic action or 
            a distribution over actions.
        """
        action, _states = self.model.predict(obs, deterministic=deterministic)
        return action, _states

    def eval_performance(self, env, side_channel, n_episodes=400, n_steps=100):
        """
        Perform empirical evaluation of the performance of the learned controller.

        Inputs
        ------
        env : Gym Environment object
            The environment in which to evaluate the controller's performance.
        side_channel : CustomSideChannel object
            An observable object that receives methods from the unity
            environment and notifies all subscribed observers.
        n_episodes : int
            Number of episodes to rollout for evaluation.
        n_steps : int
            Length of each episode.
        """

        # Set the environment to the appropriate subtask
        sub_task_string = '{},{}'.format(self.controller_ind, self.controller_ind)
        side_channel.send_string(sub_task_string)

        avg_num_steps = 0
        trials = 0
        total_steps = 0
        num_steps = 0

        # Instantiate an observer to keep track of the number of times
        # the (sub) task is successfully completed.
        observer = ObserverIncrementTaskSuccessCount(side_channel)

        for episode_ind in range(n_episodes):
            trials = trials + 1
            avg_num_steps = (avg_num_steps + num_steps) / 2

            obs = env.reset()
            num_steps = 0
            for step_ind in range(n_steps):
                num_steps = num_steps + 1
                total_steps = total_steps + 1
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)

                if done:
                    break

        # Unsubscribe the observer from the side-channel to prevent it from
        # continuing to count after the test is done.
        side_channel.unsubscribe(observer)

        # Save the resulting data
        self.data['performance_estimates'][self.data['total_training_steps']] = {
            'success_count' : observer.success_count,
            'success_rate' : observer.success_count / trials,
            'num_trials' : trials,
            'avg_num_steps' : avg_num_steps,
        }

    # def is_task_complete(self, obs):
    #     """
    #     Return true if the current observation indicates the agent has already reached its goal.
    #     """
    #     current_state = (obs[0], obs[1], obs[2])
    #     if current_state in self.final_states:
    #         return True
    #     else:
    #         return False

    def save(self, save_dir):
        """
        Save the controller object.

        Inputs
        ------
        save_dir : string
            Absolute path to the directory that will be used to save this controller.
        """
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        model_file = os.path.join(save_dir, 'model')
        self.model.save(model_file)
        controller_file = os.path.join(save_dir, 'controller_data.p')

        controller_data = {
            'controller_ind' : self.controller_ind,
            'env_settings' : self.env_settings,
            'verbose' : self.verbose,
            'max_training_steps' : self.max_training_steps,
            'data' : self.data,
        }

        with open(controller_file, 'wb') as pickleFile:
            pickle.dump(controller_data, pickleFile)

    def load(self, env, save_dir):
        """
        Load a controller object

        Inputs
        ------
        env : Gym Environment object
            The environment in which the controller will be acting.
        save_dir : string
            Absolute path to the directory that will be used to save this controller.
        """

        controller_file = os.path.join(save_dir, 'controller_data.p')
        with open(controller_file, 'rb') as pickleFile:
            controller_data = pickle.load(pickleFile)

        self.controller_ind = controller_data['controller_ind']
        self.env_settings = controller_data['env_settings']
        self.max_training_steps = controller_data['max_training_steps']
        self.verbose = controller_data['verbose']
        self.data = controller_data['data']

        model_file = os.path.join(save_dir, 'model')
        self.model = PPO.load(model_file, env=env)

    def get_success_prob(self):
        # Return the most recently estimated probability of success
        max_total_training_steps = np.max(list(self.data['performance_estimates'].keys()))
        return np.copy(self.data['performance_estimates'][max_total_training_steps]['success_rate'])

    # def _set_training_env(self, env_settings):
    #     self.training_env = Maze(**env_settings)
    #     self.training_env.agent_start_states = self.init_states
    #     self.training_env.goal_states = self.final_states

    def _init_learning_alg(self, env, verbose=False):
        self.model = PPO("MlpPolicy", 
                            env, 
                            verbose=verbose,
                            n_steps=512,
                            batch_size=64,
                            gae_lambda=0.95,
                            gamma=0.99,
                            n_epochs=10,
                            ent_coef=0.0,
                            learning_rate=2.5e-4,
                            clip_range=0.2)

    def demonstrate_capabilities(self, 
                                    env, 
                                    side_channel,
                                    n_episodes=5, 
                                    n_steps=100, 
                                    render=True):
        """
        Demonstrate the capabilities of the learned controller in the 
        environment used to train it.

        Inputs
        ------
        env : Gym Environment object
            The environment in which to evaluate the controller's performance.
        side_channel : CustomSideChannel object
            An observable object that receives methods from the unity
            environment and notifies all subscribed observers.
        n_episodes : int
            Number of episodes to rollout for evaluation.
        n_steps : int
            Length of each episode.
        render (optional) : bool
            Whether or not to render the environment at every timestep.
        """
        # Set the environment to the appropriate subtask
        sub_task_string = '{},{}'.format(self.controller_ind, self.controller_ind)
        side_channel.send_string(sub_task_string)

        for episode_ind in range(n_episodes):
            obs = env.reset()
            for step in range(n_steps):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if done:
                    break

        obs = env.reset()