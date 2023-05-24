import sys, os
sys.path.append('..')
from typing import Optional
from controllers.unity_controller import UnityController
import yaml

def load_env_info(env_info_file_name):
    """
    Load the environment information yaml file specifying the HLMDP structure.

    Parameters
    ----------
    env_info_file_name : str
        Name of the yaml file containing the environment information.
    
    Returns
    -------
    env_info : dict
        Dictionary containing the environment information.
    """
    env_info_folder = os.path.abspath('../environments')
    env_info_str = os.path.join(env_info_folder, env_info_file_name)
    with open(env_info_str, 'rb') as f:
        env_info = yaml.safe_load(f)
    return env_info

def instantiate_controllers(
        env, 
        env_settings: dict,
        verbose : Optional[bool] = None,
        num_controllers : Optional[int] = None,
        load_dir : Optional[str] = None,
        pre_trained_load_dir : Optional[str] = None,
        tensorboard_log : Optional[str] = None,
):
    """
    Instantiate a list of controllers for the environment.
    
    Parameters
    ----------
    env : UnityEnvironment
        The Unity environment.
    env_settings : dict
        Dictionary of environment settings.
    verbose : bool, optional
        Whether to print extra information. The default is True.
        This can only be set when instantiating new policies, not 
        when loading old ones.
    num_controllers : int, optional
        Number of controllers to instantiate. The default is None.
        Only set this if you want to instantiate new controllers.
    load_dir : str, optional
        Path to directory containing saved controllers. The default is None.
        Only set this if you want to load previously trained controllers.
    pre_trained_load_dir : str, optional
        Path to directory containing pre-trained controller. The default is None.
        Only set this if you want to load a pre-trained controller as the starting
        policy for all of the controllers.
    tensorboard_log : str, optional
        Path to directory to save tensorboard logs. The default is None and
        no tensorboard logs are saved. This can only be set when instantiating new policies, not 
        when loading old ones.
    """
    controller_list = []

    # Create or load the list of partially instantiated subtask controllers
    if (num_controllers is not None) and (load_dir is None) and (pre_trained_load_dir is None):
        for i in range(num_controllers):
            if tensorboard_log is None:
                tensorboard_logdir = None
            else:
                tensorboard_logdir = os.path.join(tensorboard_log , 'controller_{}'.format(i))

            controller_list.append(
                UnityController(
                    i, 
                    env, 
                    env_settings=env_settings, 
                    verbose=verbose,
                    tensorboard_log=tensorboard_logdir,
                )
            )

    elif (num_controllers is None) and (load_dir is not None) and (pre_trained_load_dir is None):
        for controller_dir in os.listdir(load_dir):
            controller_load_path = os.path.join(load_dir, controller_dir)
            if os.path.isdir(controller_load_path):
                if tensorboard_log is None:
                    tensorboard_logdir = None
                else:
                    tensorboard_logdir = os.path.join(tensorboard_log, controller_dir)
                controller = UnityController(
                    0, 
                    env, 
                    load_dir=controller_load_path, 
                    verbose=verbose,
                    tensorboard_log=tensorboard_logdir,
                )
                controller_list.append(controller)

        # re-order the controllers by index
        reordered_list = []
        for i in range(len(controller_list)):
            for controller in controller_list:
                if controller.controller_ind == i:
                    reordered_list.append(controller)
        controller_list = reordered_list

    elif (num_controllers is not None) and (load_dir is None) and (pre_trained_load_dir is not None):
        if 'pretrained_controller' not in pre_trained_load_dir:
            pre_trained_load_dir = os.path.abspath(os.path.join(pre_trained_load_dir, 'pretrained_controller'))

        for i in range(num_controllers):
            if tensorboard_log is None:
                tensorboard_logdir = None
            else:
                tensorboard_logdir = os.path.join(tensorboard_log, 'controller_{}'.format(i))

            controller = UnityController(
                0, 
                env, 
                load_dir=pre_trained_load_dir,
                verbose=verbose,
                tensorboard_log=tensorboard_logdir,
            )
        
            init_data = {
                'pre_training_steps' : controller.data['total_training_steps'],
                'total_training_steps' : 0,
                'performance_estimates' : {},
                'required_success_prob' : 0,
            }

            controller.controller_ind = i
            controller.env_settings = env_settings
            controller.data = init_data

            controller_list.append(controller)

    else:
        raise ValueError('Must specify either: num_controllers, num_controllers and pre_trained_load_dir, or load_dir.')

    return controller_list


if __name__ == '__main__':
    
    from environments.unity_env import build_unity_env
    from controllers.unity_controller import UnityController
    import os, sys
    import yaml

    from examples.gq_robotics.config.gq_20_subgoals_config import cfg

    env_info = load_env_info(cfg['hlmdp_file_name'])

    pre_trained_load_dir = os.path.abspath(os.path.join('..', 'examples', 'gq_robotics', 'data', 'saved_controllers', '2023-05-20_16-32-24_pretrain_warthog_controller'))

    env_settings = {
        'time_scale' : 99.0,
    }

    env, side_channels = build_unity_env()
    side_channels['engine_config_channel'].set_configuration_parameters(
                                            time_scale=env_settings['time_scale'])
    
    tensorboard_log = 'tensorboard_controller_logs/'

    # Instantiate the controllers
    controller_list = instantiate_controllers(env, env_settings, num_controllers=10, pre_trained_load_dir=pre_trained_load_dir, tensorboard_log=tensorboard_log)
    # controller_list = instantiate_controllers(env, env_settings, num_controllers=10, verbose=False)

    print([controller_list[i].data for i in range(len(controller_list))])

    print(controller_list[0].verbose)

    controller_list[0].learn(side_channels['custom_side_channel'], 
                                total_timesteps=1000)

    env.close()