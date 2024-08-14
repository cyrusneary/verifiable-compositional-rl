# Verifiable Compositional Reinforcement Learning Systems

This code implements a novel framework for verifiable and compositional reinforcement learning systems.

## Installation

### Prerequisites
1. Install Gurobi [https://www.gurobi.com/downloads/](https://www.gurobi.com/downloads/)
	- Academic Gurobi licenses may be requested from [https://www.gurobi.com/downloads/end-user-license-agreement-academic/](https://www.gurobi.com/downloads/end-user-license-agreement-academic/)
2. Install Unity Hub & Setup
	- Unity 2020.3.20f1 with Mlagents package v2.1.0.
    	- Begin by installing the Unity Hub. Here is the [download link](https://unity.com/download#how-get-started).
    	- Next use the Unity Hub to install the appropriate version of Unity (2020.3.20f1).
    	- Finally, drag the entire unity_labyrinth folder structure (which contains the unity labyrinth environment files) into the project list on Unity Hub. Use the Unity Hub to open the newly created project in order to get started.

### Step by step instructions
1. Download & install [miniforge](https://github.com/conda-forge/miniforge)
2. Open a terminal window and enter: `git clone `
3. In a terminal navigate to the directory this README.md is in
4. Go to https://utexas.box.com/s/6tqfxf3mb0eg7qwni85qeia82z34paqw to download and install the .tar ball that is necessary to recreate the environment required to run the repository. The name of the file should be `comp_rl.tar.gz`. Place this file in the base `~/verifiable-compositional-rl/` directory.
5. `mkdir comp_rl`
6.  `sudo tar -xzvf comp_rl.tar.gz -C /comp_rl`
7.  `source ~/verifiable-compositional-rl/comp_rl/bin/activate`
12. `cd src/examples`
13. `python test_minigrid.py`

- If the steps above run successfully, you are ready to run and train models using the repository 

### Running the Unity Labyrinth training scripts.

Other training scripts that can be run under ‘src/examples` are:

run_unity_labyrinth.py: It initializes the simulation on the Unity Labyrinth environment provided in the prerequisites. If the prior steps have been achieved successfully, run the script and the command window shall instruct you to go to Unity and press play to start the scene and the training. Once training is complete (which can take 15+ minutes, depending on your hardware configuration) 
      





### Python dependencies
This library uses the following packages:
- matplotlib
- numpy
- pickle
- gurobipy [https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_grbpy_the_gurobi_python.html](https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_grbpy_the_gurobi_python.html)
- gym_minigrid [https://github.com/maximecb/gym-minigrid](https://github.com/maximecb/gym-minigrid)
- gym [https://github.com/openai/gym](https://github.com/openai/gym)
- stable_baselines3 [https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- pytorch [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
 	- Note: pytorch will be installed automatically by stable_baselines3 if you do not already have it installed.
- mlagents v0.27.0 (release 18) [https://github.com/Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)


## Replicating experiments

### Discrete Gridworld Labyrinth Experiments
To run the example presented in the paper, navigate to the src directory and run:
> python run_minigrid_labyrinth.py

This will setup and run the entire labyrinth experiment, and it will automatically generate
new folders in which to save the learned sub-systems and the testing results within src/data.

### Continuous Labyrinth Experiments
To run the continuous Labyrinth experiments, some extra setup is required.
- First, download the appropriate version of Unity as instructed above.
- Next, grab the Unity labyrinth project folder from [this repository](https://github.com/cyrusneary/unity_labyrinth), and drop it into the Unity HUB to create a new version of the labyrinth project in Unity.
- Open this labyrinth project and in the Unity editor, open scenes\SampleScene.
- In *this* repository, navigate to the \src directory. Run:
	> python run_unity_labyrinth.py
- You should see the line "Listening on port 5004. Start training by pressing the Play button in the Unity Editor."
- Click on the play button at the top of the Unity editor to begin training.

### Inverse reinforcement learning scripts 

Do not run the inverse reinforcement learning scripts (labeled with ‘irl’ in their names). They’re disabled/unusable as of the time of this writing.


### Data from trained models

For either the minigrid or unity labyrinth scripts, a folder named ‘YYYY-MM-DD_HH_MiMi_SS_unity/minigrid_labyrinth’ will be generated under the ‘~/src/data/saved_controllers’ directory. Within the name, YYYY-MM-DD corresponds to the year, month, and day the simulation was run, HH_MiMi_SS is the time of the day down to the seconds when the simulation was run, and either unity/minigrid is the experiment that was run. (e.g: 2024-07-25_14-02-58_unity_labyrinth). Each folder includes data regarding the training details of each controller to achieve the simulation results

Keep track of this folder name, as it will be essential to run the plotting scripts later on.


### Steps to run plotting and visualization scripts

Go to `~/src/plotting-visualtization`, where you’ll find the curstomized plotting and visualization scripts. Ignore `~/src/plotting`, as it's just there to allow the main training scripts to run appropriately.

# Plotting scripts

The plotting scripts are as follows:

`plot_training_results.py`: Plots the training results for a given training folder
`plot_training_schedule.py`: Plots the training schedule resulting from a given training folder
`plot_sub_task_specifications.py`: Plots the subtask specifications for a given training folder
`plot_training_variance.py`: Plots the training variance between N many given training folders.

To use any of these scripts, run them using the following syntax on the terminal window:

`python script.py experiment_name folder_name`

Where experiment_name is either unity_labyrinth or minigrid_labyrinth depending on what environment the training folder corresponds to. For example, for the training results script you could run:

`python plot_training_results.py unity_labyrinth 2024-07-25_14-02-58_unity_labyrinth`

Which will instantly show a plot of the plotted simulation on your screen (which can be closed doing ctrl+C on the command window) and save a .png and .tex copy of the plotted image on the  ‘~/src/plotting/figures’ folder.

The remainder of the scripts work the same, with the exception of plot_training_variance.py, which takes in one or more training folders in the form, for as many folders as you’d like to test. Note that this is the only script in the repo designed to take in more than one folder: 

`python plot_training_variance.py experiment_name folder1 folder2 folder3 … folderN-1 folderN`

For example:

`python plot_training_variance.py minigrid_labyrinth 2021-05-26_22-31-53_minigrid_labyrinth 2021-05-26_22-32-00_minigrid_labyrinth 2021-05-26_22-32-07_minigrid_labyrinth`



# Visualization scripts

The following are the visualization scripts, which show the user the final training results/trajectory picked resulting from the simulations. The scripts are as follows:

`visualize_unity_labyrinth_controllers.py`: It will show you the results of the Unity labyrinth experiment for a given folder. Upon running the script for its proper folder, you’ll be instructed to press play on the Unity scene where you have the labyrinth environment. Once you do so, you should be able to see the path resulting from your training controller.

`visualize_gridworld_labyrinth_controllers.py`: Disabled

`visualize_gridworld_pixel_labyrinth_controllers.py`: Disabled

To run any of these scripts, follow the following format:

python script.py folder_name

For example: `python visualize_unity_labyrinth_controllers.py 2021-12-14_10-26-12_unity_labyrinth`



As inverse reinforcement learning capabilities are not enabled in this repository, please do not use `plot_irl_results.py`  or `visualize_irl_unity_labyrinth_controllers.py`.

                         

## Citation
To cite this project, please use:
~~~
@inproceedings{neary2022verifiable,
  title={Verifiable and compositional reinforcement learning systems},
  author={Neary, Cyrus and Verginis, Christos and Cubuktepe, Murat and Topcu, Ufuk},
  booktitle={Proceedings of the International Conference on Automated Planning and Scheduling},
  volume={32},
  pages={615--623},
  year={2022}
}
~~~
