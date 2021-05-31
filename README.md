# Verifiable Compositional Reinforcement Learning Systems

This code implements a novel framework for verifiable and compositional reinforcement learning systems.

## Installation

To run the code, Python 3.8 is required with the following packages:

    - matplotlib
    - numpy
    - pickle
    - gurobipy (https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_grbpy_the_gurobi_python.html)
    - gym_minigrid (https://github.com/maximecb/gym-minigrid)
    - gym (https://github.com/openai/gym)
    - stable_baselines3 (https://github.com/DLR-RM/stable-baselines3)
        - pytorch (https://pytorch.org/get-started/locally/)
            - Note: pytorch will be installed automatically by stable_baselines3
            if you do not already have it installed.

Gurobi optimization software is also necessary. This can be downloaded from (https://www.gurobi.com/downloads/).
Academic Gurobi licenses may be requested from (https://www.gurobi.com/downloads/end-user-license-agreement-academic/).

## Running examples

To run the example presented in the paper, run 
    src>>python3 run_minigrid_labyrinth.py

This will setup and run the entire labyrinth experiment, and it will automatically generate 
new folders in which to save the learned sub-systems and the testing results within src/data.