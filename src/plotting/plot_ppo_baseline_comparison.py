# %%
from datetime import time
from pickle import load
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os, sys
sys.path.append('..')

from utils.results_saver import Results

import tikzplotlib

# %%
load_folder_name = '2021-05-19_16-25-50_minigrid_labyrinth_baseline_one_component'
load_folder_name = '2021-05-19_21-36-11_minigrid_labyrinth_baseline_one_component'

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

ax.plot([x[0], x[-1]], [required_success_prob, required_success_prob],
        color='black',
        linewidth=large_linewidth,
        linestyle='--',
        label='Required Probability of Success')
ax.plot(x, comp_empirical_success, 
        color='black',
        marker='d',
        markersize=large_marker_size,
        linewidth=large_linewidth,
        label='Empirically Measured Probability of Task Success')
ax.legend(fontsize=15)

yl = ax.get_ylim()
ax.set_ylim(yl)

# save_path = os.path.join(os.path.curdir, 'figures', 'training_curves.tex')
# tikzplotlib.save(save_path)

# %%
