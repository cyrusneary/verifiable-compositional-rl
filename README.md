# Verifiable Compositional Reinforcement Learning Systems

This code implements a novel framework for verifiable and compositional reinforcement learning systems. 

## Installation

### Python requirements
The most simple installation procedure is to use Anaconda. 
- First, download Anaconda3 from [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual).
- Next, navigate to this directory and run:
    > conda env create -f environment.yml

If you wish to manually install the necessary packages, Python 3.8 is required with the following packages:

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

### Other requirements

- Gurobi optimization software. This can be downloaded from [https://www.gurobi.com/downloads/](https://www.gurobi.com/downloads/).
Academic Gurobi licenses may be requested from [https://www.gurobi.com/downloads/end-user-license-agreement-academic/](https://www.gurobi.com/downloads/end-user-license-agreement-academic/).

- Unity 2020.3.20f1 with Mlagents package v2.1.0.
    - Begin by installing the Unity Hub. Here is the [download link](https://unity.com/download#how-get-started).
    - Next use the Unity Hub to install the appropriate version of Unity (2020.3.20f1).
    - Finally, drag the entire unity_labyrinth folder structure (which contains the unity labyrinth environment files) into the project list on Unity Hub. Use the Unity Hub to open the newly created project in order to get started.

## Running examples

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
    >python run_unity_labyrinth.py
- You should see the line "Listening on port 5004. Start training by pressing the Play button in the Unity Editor."
- Click on the play button at the top of the Unity editor to begin training.

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
