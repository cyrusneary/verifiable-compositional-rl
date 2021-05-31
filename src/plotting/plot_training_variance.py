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
load_folder_name_list = [
    '2021-05-26_22-31-53_minigrid_labyrinth',
    '2021-05-26_22-32-00_minigrid_labyrinth',
    '2021-05-26_22-32-07_minigrid_labyrinth',
    '2021-05-26_22-32-12_minigrid_labyrinth',
    '2021-05-26_22-32-46_minigrid_labyrinth',
    '2021-05-26_22-33-16_minigrid_labyrinth',
    '2021-05-26_22-33-42_minigrid_labyrinth',
    '2021-05-26_22-34-26_minigrid_labyrinth',
    '2021-05-26_22-34-49_minigrid_labyrinth',
    '2021-05-26_22-35-16_minigrid_labyrinth',
    ]

results_dict_list = []

experiment_name = 'minigrid_labyrinth'

base_path = os.path.abspath(os.path.curdir)
string_ind = base_path.find('/src')
assert(string_ind >= 0)
base_path = base_path[0:string_ind + 4]
base_path = os.path.join(base_path, 'data', 'saved_controllers')

for load_folder_name in load_folder_name_list:
    load_dir = os.path.join(base_path, load_folder_name)
    results_dict_list.append(Results(load_dir=load_dir))

# %%
required_success_prob = results_dict_list[0].data['prob_threshold']

timesteps = results_dict_list[0].data['cparl_loop_training_steps']

comp_pred_success_median = []
comp_pred_success_75 = []
comp_pred_success_25 = []

for t in timesteps:
    temp = []
    for results_dict in results_dict_list:
        if t in results_dict.data['composition_predicted_success_prob'].keys():
            temp.append(results_dict.data['composition_predicted_success_prob'][t])
        else:
            temp.append(1.0)
    
    comp_pred_success_median.append(np.percentile(temp, 50))
    comp_pred_success_75.append(np.percentile(temp, 75))
    comp_pred_success_25.append(np.percentile(temp, 25))

comp_pred_success_median.insert(0,0)
comp_pred_success_75.insert(0,0)
comp_pred_success_25.insert(0,0)

comp_empirical_success_median = []
comp_empirical_success_75 = []
comp_empirical_success_25 = []

for t in timesteps:
    temp = []
    for results_dict in results_dict_list:
        if t in results_dict.data['composition_rollout_mean'].keys():
            temp.append(results_dict.data['composition_rollout_mean'][t])
        else:
            temp.append(1.0)
    
    comp_empirical_success_median.append(np.percentile(temp, 50))
    comp_empirical_success_75.append(np.percentile(temp, 75))
    comp_empirical_success_25.append(np.percentile(temp, 25))

comp_empirical_success_median.insert(0,0)
comp_empirical_success_75.insert(0,0)
comp_empirical_success_25.insert(0,0)

x = timesteps.copy()
x.insert(0,0)

# %%
small_linewidth = 3
small_marker_size = 10

large_linewidth = 5
large_marker_size=20

alpha = 0.1

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.grid()

ax.plot([x[0], x[-1]], [required_success_prob, required_success_prob],
        color='black',
        linewidth=large_linewidth,
        linestyle='--',
        label='Required Probability of Success')

ax.plot(x, comp_pred_success_median, 
        color='blue',
        marker='d',
        markersize=large_marker_size,
        linewidth=large_linewidth,
        label='Lower Bound on Probability of Task Success')
ax.fill_between(x, comp_pred_success_25, comp_pred_success_75,
                color='blue',
                alpha=alpha)

ax.plot(x, comp_empirical_success_median, 
        color='black',
        marker='d',
        markersize=large_marker_size,
        linewidth=large_linewidth,
        label='Empirically Measured Probability of Task Success')
ax.fill_between(x, comp_empirical_success_25, comp_empirical_success_75,
                color='black',
                alpha=alpha)

yl = ax.get_ylim()
ax.set_ylim(yl)

ax.legend(fontsize=15)

save_path = os.path.join(os.path.curdir, 'figures', 'variance_training_curves.tex')
tikzplotlib.save(save_path)

# %%

# %%
