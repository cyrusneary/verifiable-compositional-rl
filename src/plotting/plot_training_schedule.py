# %%
from datetime import time
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os, sys
sys.path.append('..')

from utils.results_saver import Results
import tikzplotlib

# %%
# load_folder_name = '2021-05-22_13-53-56_minigrid_labyrinth'
# experiment_name = 'minigrid_labyrinth'
load_folder_name = '2021-12-13_22-26-40_unity_labyrinth'
load_folder_name = '2022-10-14_19-10-42_unity_labyrinth'
load_folder_name = '2022-10-23_12-27-58_unity_labyrinth'
experiment_name = 'unity_labyrinth'

base_path = os.path.abspath(os.path.curdir)
string_ind = base_path.find('src')
assert(string_ind >= 0)
base_path = base_path[0:string_ind + 4]
base_path = os.path.join(base_path, 'data', 'saved_controllers')

load_dir = os.path.join(base_path, load_folder_name)

results = Results(load_dir=load_dir)

# %%
required_success_prob = results.data['prob_threshold']

timesteps = results.data['cparl_loop_training_steps']
controller_training_steps = results.data['controller_elapsed_training_steps']

# %%

small_linewidth = 3
small_marker_size = 10

large_linewidth = 20
large_marker_size = 50

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.grid()
ax.set_yticks([i for i in controller_training_steps.keys()])

ttwenty = cm.get_cmap('tab20')

for t in range(len(timesteps) - 1):
    curr_time = timesteps[t]
    next_time = timesteps[t+1]
    for controller_ind in controller_training_steps.keys():
        if not controller_training_steps[controller_ind][curr_time] == controller_training_steps[controller_ind][next_time]:
            ax.plot([curr_time, next_time], [controller_ind, controller_ind],
                    linewidth=large_linewidth,
                    color=ttwenty(controller_ind))

yl = ax.get_ylim()
ax.set_ylim(yl)

ax.plot([0.25e6, 0.25e6], [yl[0],yl[1]],
        color='red',
        linewidth=large_linewidth*0.7,
        linestyle='--')
        
save_path = os.path.join(os.path.curdir, 'figures', experiment_name + '_training_schedule.tex')
tikzplotlib.save(save_path)


# %%

# %%
