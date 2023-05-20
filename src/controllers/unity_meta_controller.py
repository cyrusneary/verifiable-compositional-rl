import numpy as np
import sys
sys.path.append('..')
from environments.unity_env import CustomSideChannel
from MDP.general_high_level_mdp import HLMDP
from utils.observers import ObserverIncrementTaskSuccessCount

class MetaController(object):

    """
    Object representing a meta controller.
    This is the software object representing the composite system, comprising
    of a meta-policy and a list of implemented sub-systems.
    """

    def __init__(self, 
                meta_policy: np.ndarray,
                hlmdp: HLMDP,
                side_channels: dict):
        """
        Inputs
        ------
        meta_policy
            Numpy array representing the meta-policy.
        hlmdp
            The high-level MDP in which this meta-controller is acting.
        side_channels
            Dictionary of side channel objects used to specify environment
            settings before running.
        """
        self.meta_policy = meta_policy
        self.controller_list = hlmdp.controller_list
        self.controller_indeces = np.arange(len(self.controller_list))
        self.current_controller_ind = None
        self.successor = hlmdp.successor

        self.s_i = hlmdp.s_i
        self.s_g = hlmdp.s_g
        side_channels['custom_side_channel'].subscribe(self)

        self.reset(side_channels)

    def notify(self, observable: CustomSideChannel, message: str):
        """
        Receive messages from the side information channel, and update the 
        abstract state and the current sub-task accordingly.

        Inputs
        ------
        observable
            An observable side channel which acts as the observable, notifying
            this observer each time a new message is received from the 
            environment.
        messasge
            A string containing the message passed on by the observable.
        """
        if message == 'Completed sub task: {}'.format(self.current_controller_ind):
            # Get the new abstract state
            self.current_abstract_state = \
                self.successor[(self.current_abstract_state, 
                                self.current_controller_ind)]

            if not(self.current_abstract_state == self.s_g):
                # Get the next abstract action to take
                self.current_controller_ind = \
                    self.select_next_abstract_action(self.current_abstract_state)

                observable.send_string('-1,{}'.format(self.current_controller_ind))
                # print('Current controller: {}'.format(self.current_controller_ind))

        elif message == 'Failed task':
            pass
        elif message == 'Completed task':
            pass
        elif message == '':
            pass
        else:
            pass
            # raise Exception('Unexpected message received from unity environment.')

    def unsubscribe_meta_controller(self, side_channels: dict):
        """
        A function that unsubscribes the current meta-controller to prevent it
        from continuing to send messages to the unity enviroment.
        
        Inputs
        ------
        side_channels
            Dictionary of side channel objects used to specify environment
            settings before running.
        """
        side_channels['custom_side_channel'].unsubscribe(self)

    def reset(self, side_channels):
        self.current_abstract_state = self.s_i
        self.current_controller_ind = self.select_next_abstract_action(
                                            self.current_abstract_state)
        side_channels['custom_side_channel'].send_string('-1,{}'.format(self.current_controller_ind))
        # print('Current controller: {}'.format(self.current_controller_ind))

    def select_next_abstract_action(self, abstract_state):
        """
        Inputs
        ------
        abstact_state : int
            Integer representation of the current abstract state.
        
        Outputs
        -------
        abstract_action : int
            Integer representation of the next abstract action to take.
        """
        return np.random.choice(len(self.controller_list), 
                                p=self.meta_policy[abstract_state, :])

    def predict(self, obs, deterministic=True):
        """
        Get the system's action, given the current environment observation (state)

        Inputs
        ------
        obs : tuple
            Tuple representing the current environment observation (state).
        deterministic (optional) : bool
            Flag indicating whether or not to return a deterministic action or 
            a distribution over actions.
        """
        # Grab the currently selected controller
        controller = self.controller_list[self.current_controller_ind]
        action, _states = controller.predict(obs, deterministic=deterministic)

        return action, _states

    def eval_performance(self, env, side_channels, n_episodes=200, n_steps=1000):
        """
        Perform empirical evaluation of the performance of the meta controller.

        Inputs
        ------
        env : environment
            Environment to perform evaluation in.
        side_channels : dict
            Dictionary of side channel objects used to specify environment
            settings before running.
        n_episodes (optional) : int
            Number of episodes to rollout for evaluation.
        n_steps (optional) : int
            Length of each episode.

        Outputs
        -------
        success_rate : float
            Empirically measured rate of success of the meta-controller.
        """
        # Set the environment task to be the overall composite task
        side_channels['custom_side_channel'].send_string('-1,-1')

        avg_num_steps = 0
        trials = 0
        total_steps = 0
        num_steps = 0

        # Instantiate an observer to keep track of the number of times
        # the (sub) task is successfully completed.
        observer = ObserverIncrementTaskSuccessCount(side_channels['custom_side_channel'])

        for episode_ind in range(n_episodes):
            trials = trials + 1
            avg_num_steps = (avg_num_steps + num_steps) / 2

            obs = env.reset()
            self.reset(side_channels)

            num_steps = 0
            for step_ind in range(n_steps):
                num_steps = num_steps + 1
                total_steps = total_steps + 1
                action, _states = self.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if done:
                    break

        # Unsubscribe the observer from the side-channel to prevent it from
        # continuing to count after the test is done.
        side_channels['custom_side_channel'].unsubscribe(observer)

        return observer.success_count / trials

    def demonstrate_capabilities(self, 
                                env, 
                                side_channels,
                                n_episodes=5, 
                                n_steps=200, 
                                render=True):
        """
        Run the meta-controller in an environment and visualize the results.

        Inputs
        ------
        env : Minigrid gym environment
            Environment to perform evaluation in.
        side_channels : dict
            Dictionary of side channel objects used to specify environment
            settings before running.
        n_episodes (optional) : int
            Number of episodes to rollout for evaluation.
        n_steps (optional) : int
            Length of each episode.
        render (optional) : bool
            Flag indicating whether or not to render the environment.
        """
        # Set the timescale of the simulation back to real time
        side_channels['engine_config_channel']\
            .set_configuration_parameters(time_scale=1.0)

        # Set the environment task to be the overall composite task
        side_channels['custom_side_channel'].send_string('-1,-1')

        for episode_ind in range(n_episodes):
            obs = env.reset()
            self.reset(side_channels)
            for step in range(n_steps):
                action, _states = self.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if done:
                    print('Episode {} ended after {} steps.'.format(episode_ind, step))
                    break

