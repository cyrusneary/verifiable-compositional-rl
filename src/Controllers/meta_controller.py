import numpy as np

class MetaController(object):

    """
    Object representing a meta controller.
    This is the software object representing the composite system, comprising
    of a meta-policy and a list of implemented sub-systems.
    """

    def __init__(self, meta_policy, controller_list, state_list):
        """
        Inputs
        ------
        meta_policy : numpy array
            Numpy array representing the meta-policy.
        controller_list : list
            List of MinigridController objects.
        state_list : list
            List of HLM states. Each of these high-level states is itself
            a list of low-level environment states.
        """
        self.meta_policy = meta_policy
        self.controller_list = controller_list
        self.controller_indeces = np.arange(len(controller_list))
        self.state_list = state_list
        self.current_controller_ind = None

    def obs_mapping(self, obs):
        """
        Map from an environment observation (state) to the corresponding 
        high-level state.

        Inputs
        ------
        obs : tuple
            Tuple representing the current environment observation (state).
        """
        state = (obs[0], obs[1], obs[2])

        obs_in_abstract_state = False

        for env_state_set in self.state_list:
            if isinstance(env_state_set, list) and state in env_state_set:
                high_level_state = self.state_list.index(env_state_set)
                obs_in_abstract_state = True
        
        if not obs_in_abstract_state:
            raise RuntimeError("Trying to enact meta-policy from state that doesn't exist in high-level state space.")

        return high_level_state

    def reset(self):
        self.current_controller_ind = None

    def predict(self, obs, deterministic=True):
        """
        Get the system's action, given the current environment observation (state)

        Inputs
        ------
        obs : tuple
            Tuple representing the current environment observation (state).
        deterministic (optional) : bool
            Flag indicating whether or not to return a deterministic action or a distribution
            over actions.
        """

        if self.current_controller_ind is not None:
            # Grab the currently selected controller
            controller = self.controller_list[self.current_controller_ind]

            # If the controller's task has been completed, deselect it
            if controller.is_task_complete(obs):
                self.current_controller_ind = None

        # In no controller is selected, choose which controller to execute
        if self.current_controller_ind is None:
            meta_state = self.obs_mapping(obs)
            controller_probabilities = self.meta_policy[meta_state, :]
            self.current_controller_ind = np.random.choice(len(self.controller_list), p=controller_probabilities)
        
        # Grab the currently selected controller
        controller = self.controller_list[self.current_controller_ind]

        action, _states = controller.predict(obs, deterministic=deterministic)

        return action, _states

    def eval_performance(self, env, n_episodes=200, n_steps=1000):
        """
        Perform empirical evaluation of the performance of the meta controller.

        Inputs
        ------
        env : Minigrid gym environment
            Environment to perform evaluation in.
        n_episodes (optional) : int
            Number of episodes to rollout for evaluation.
        n_steps (optional) : int
            Length of each episode.

        Outputs
        -------
        success_rate : float
            Empirically measured rate of success of the meta-controller.
        """
        success_count = 0
        avg_num_steps = 0
        trials = 0
        total_steps = 0
        num_steps = 0

        for episode_ind in range(n_episodes):
            trials = trials + 1
            avg_num_steps = (avg_num_steps + num_steps) / 2

            obs = env.reset()
            self.reset()

            num_steps = 0
            for step_ind in range(n_steps):
                num_steps = num_steps + 1
                total_steps = total_steps + 1
                action, _states = self.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if done:
                    if info['task_complete']:
                        success_count = success_count + 1
                    break

        return success_count / trials

    def demonstrate_capabilities(self, env, n_episodes=5, n_steps=200, render=True):
        """
        Run the meta-controller in an environment and visualize the results.

        Inputs
        ------
        env : Minigrid gym environment
            Environment to perform evaluation in.
        n_episodes (optional) : int
            Number of episodes to rollout for evaluation.
        n_steps (optional) : int
            Length of each episode.
        render (optional) : bool
            Flag indicating whether or not to render the environment.
        """
        for episode_ind in range(n_episodes):
            obs = env.reset()
            self.reset()
            for step in range(n_steps):
                action, _states = self.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if render:
                    env.render(highlight=False)
                if done:
                    break

