import numpy as np
import pickle
import os

class Results(object):
    """
    Object to save data throughout training of compositional RL systems.
    """

    def __init__(self, 
                controller_list=None, 
                env_settings=None, 
                prob_threshold=None, 
                training_iters=None, 
                estimation_rollouts=None,
                random_seed=None, 
                load_dir=None):
        """
        Either all inputs should be specified except for load_dir, 
        or load_dir must be specified.

        Inputs
        ------
        controller_list (optional) : list
            list of MinigridController objects
        env_settings (optional) : dict
            Dictionary containing the environment settings
        prob_threshold (optional) : float
            The required probability of overall task success
        training_iters (optional) : int
            The number of training steps to use when training
            each sub-system in the main iterative compositional
            RL loop.
        estimation_rollouts (optional) : int
            The number of rollouts to use when empirically estimating
            (sub-)task success probabilities.
        random_seed (optional) : int
            Random seed used in the experiment.
        load_dir (optional) : str
            String pointing to the results file to load from a previous
            run of the experiment.
        """
        
        if load_dir is None:
            assert(controller_list is not None)
            assert(env_settings is not None)
            assert(prob_threshold is not None)
            assert(training_iters is not None)
            assert(estimation_rollouts is not None)
            assert(random_seed is not None)
            self.data = {} # save everything in a dictionary
            self.data['env_settings'] = env_settings
            self.data['prob_threshold'] = prob_threshold
            self.data['training_iters'] = training_iters
            self.data['estimation_rollouts'] = estimation_rollouts
            self.data['random_seed'] = random_seed

            self.data['controller_elapsed_training_steps'] = {}
            self.data['controller_rollout_mean'] = {}
            self.data['controller_num_rollouts'] = {}
            self.data['controller_required_probabilities'] = {}
            self._construct_controller_data(controller_list)

            self.data['cparl_loop_training_steps'] = []

            self.data['composition_rollout_mean'] = {}
            self.data['composition_num_rollouts'] = {}
            self.data['composition_policy'] = {}
            self.data['composition_predicted_success_prob'] = {}
        else:
            self.load(load_dir)

    def _construct_controller_data(self, controller_list):
        """
        Add the meta-data of a list of controllers to the saved results

        Inputs
        ------
        controller_list : list
            List of MiniGrid Controller objects.
        """
        for controller in controller_list:
            controller_ind = controller.controller_ind
            self.data['controller_elapsed_training_steps'][controller_ind] = {}
            self.data['controller_rollout_mean'][controller_ind] = {}
            self.data['controller_num_rollouts'][controller_ind] = {}
            self.data['controller_required_probabilities'][controller_ind] = {}

    def update_training_steps(self, training_steps):
        """
        Update the total number of elapsed training steps of the overall system.

        Inputs
        ------
        training_steps : int
            The number of elapsed training steps since the LAST call of update_training_steps()
        """
        if self.data['cparl_loop_training_steps']:
            elapsed_training_steps = self.data['cparl_loop_training_steps'][-1]
        else:
            elapsed_training_steps = 0
        self.data['cparl_loop_training_steps'].append(elapsed_training_steps + training_steps)

    def update_controllers(self, controller_list):
        """
        Use the controller list to update results data pertaining to the sub-systems.

        Inputs
        ------
        controller_list : list
            List of MinigridController objects whose results data is to be updated.
        """
        elapsed_training_steps = self.data['cparl_loop_training_steps'][-1]
        for controller in controller_list:
            controller_ind = controller.controller_ind
            self.data['controller_elapsed_training_steps'][controller_ind][elapsed_training_steps] = controller.data['total_training_steps']
            self.data['controller_rollout_mean'][controller_ind][elapsed_training_steps] = controller.data['performance_estimates'][controller.data['total_training_steps']]['success_rate']
            self.data['controller_num_rollouts'][controller_ind][elapsed_training_steps] = controller.data['performance_estimates'][controller.data['total_training_steps']]['num_trials']
            self.data['controller_required_probabilities'][controller_ind][elapsed_training_steps] = controller.data['required_success_prob']

    def update_composition_data(self, rollout_mean, num_rollouts, policy, predicted_success_prob):
        """
        Update the data pertaining to the compostional system's performance.

        Inputs
        ------
        rollout_mean : float
            Empirically estimated probability of the composite system's task success.
        num_rollouts : int
            Number of system rollouts used to estimate the rollout_mean.
        policy : numpy array
            The meta-policy specifying the composite system.
        predicted_success_prob : float
            The HLM's predicted probability of task success.
        """
        elapsed_training_steps = self.data['cparl_loop_training_steps'][-1]
        self.data['composition_rollout_mean'][elapsed_training_steps] = rollout_mean
        self.data['composition_num_rollouts'][elapsed_training_steps] = num_rollouts
        self.data['composition_policy'][elapsed_training_steps] = policy
        self.data['composition_predicted_success_prob'][elapsed_training_steps] = predicted_success_prob

    def save(self, save_dir):
        """
        Save the Results object.

        Inputs
        ------
        save_dir : string
            Absolute path to the directory that will be used to save this results data.
        """
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        data_file = os.path.join(save_dir, 'results_data.p')

        with open(data_file, 'wb') as pickleFile:
            pickle.dump(self.data, pickleFile)

    def load(self, save_dir):
        """
        Load a Results object

        Inputs
        ------
        save_dir : string
            Absolute path to the directory that will be used to save this results data.
        """
        data_file = os.path.join(save_dir, 'results_data.p')
        with open(data_file, 'rb') as pickleFile:
            results_data = pickle.load(pickleFile)

        self.data = results_data
        
    