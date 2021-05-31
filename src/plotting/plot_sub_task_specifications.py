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
load_folder_name = '2021-05-22_13-53-56_minigrid_labyrinth'

experiment_name = 'minigrid_labyrinth'

base_path = os.path.abspath(os.path.curdir)
string_ind = base_path.find('/src')
assert(string_ind >= 0)
base_path = base_path[0:string_ind + 4]
base_path = os.path.join(base_path, 'data', 'saved_controllers')

load_dir = os.path.join(base_path, load_folder_name)

results = Results(load_dir=load_dir)

# %%
required_success_prob = results.data['prob_threshold']

timesteps = results.data['cparl_loop_training_steps']
controller_required_probabilities = results.data['controller_required_probabilities']


# %%

small_linewidth = 3
small_marker_size = 10

large_linewidth = 5
large_marker_size = 50

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.grid()

ax.plot([timesteps[0], timesteps[-1]], [required_success_prob, required_success_prob],
        color='black',
        linewidth=large_linewidth,
        linestyle='--',
        label='Required Probability of Success')

ttwenty = cm.get_cmap('tab20')

for controller_ind in controller_required_probabilities.keys():
    required_probs = [controller_required_probabilities[controller_ind][t] for t in timesteps]
    ax.plot(timesteps, required_probs,
            marker='d',
            markersize=small_marker_size,
            linewidth=large_linewidth,
            color=ttwenty(controller_ind))
        
ax.plot([0.8e6, 0.8e6], [yl[0],yl[1]],
        color='red',
        linewidth=large_linewidth*2,
        linestyle='--')

save_path = os.path.join(os.path.curdir, 'figures', 'sub_task_specifications.tex')
tikzplotlib.save(save_path)

# %%

time1 = 0.6e6
time_transition = 0.8e6
time2 = 1.0e6

for controller_ind in controller_required_probabilities.keys():
    print('Controller {}, At time {}, Required sub-task probability of success: {}'.format(controller_ind, time1, controller_required_probabilities[controller_ind][time1]))
    print('COntroller {}, At time {}, Empirical performance: {}'.format(controller_ind, time1, results.data['controller_rollout_mean'][controller_ind][time1]))
    print('COntroller {}, At time {}, Empirical performance: {}'.format(controller_ind, time_transition, results.data['controller_rollout_mean'][controller_ind][time_transition]))
    print('Controller {}, At time {}, Required sub-task probability of success: {}'.format(controller_ind, time2, controller_required_probabilities[controller_ind][time2]))
# %%
