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
# load_folder_name = '2021-12-13_22-26-40_unity_labyrinth'
# load_folder_name = '2022-10-23_12-27-58_unity_labyrinth'
experiment_name = 'unity_labyrinth'
load_folder_name = '2022-10-13_22-25-00_minigrid_pixel_labyrinth'
experiment_name = 'minigrid_pixel_labyrinth'

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
comp_pred_success = [results.data['composition_predicted_success_prob'][t] for t in timesteps]
comp_pred_success.insert(0,0)
comp_empirical_success = [results.data['composition_rollout_mean'][t] for t in timesteps]
comp_empirical_success.insert(0,0)

x = timesteps.copy()
x.insert(0,0)

# %%

small_linewidth = 3
small_marker_size = 10

large_linewidth = 5
large_marker_size=20

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.grid()

ttwenty = cm.get_cmap('tab20')

for i in range(len(results.data['controller_rollout_mean'])):
    y = [results.data['controller_rollout_mean'][i][t] for t in timesteps]
    y.insert(0,0)
    ax.plot(x, y,
            linewidth=small_linewidth,
            marker='d',
            markersize=small_marker_size,
            color=ttwenty(i))
            # label='Component {} Empirical Probability of Success'.format(i))

ax.plot([x[0], x[-1]], [required_success_prob, required_success_prob],
        color='black',
        linewidth=large_linewidth,
        # linestyle='--',
        label='Required Probability of Success')
ax.plot(x, comp_pred_success, 
        color='blue',
        marker='d',
        markersize=large_marker_size,
        linewidth=large_linewidth,
        label='Lower Bound on Probability of Task Success')
ax.plot(x, comp_empirical_success, 
        color='black',
        marker='d',
        markersize=large_marker_size,
        linewidth=large_linewidth,
        label='Empirically Measured Probability of Task Success')
# ax.legend(fontsize=15)

yl = ax.get_ylim()
ax.set_ylim(yl)

ax.plot([6.5e5, 6.5e5], [yl[0],yl[1]],
        color='red',
        linewidth=large_linewidth*2,
        # linestyle='--',
        )

# plt.show()

save_path = os.path.join(os.path.curdir, 'figures', experiment_name + '_training_curves.tex')
tikzplotlib.save(save_path)


# %%

# %%
