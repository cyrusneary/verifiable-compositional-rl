import numpy as np
import matplotlib.pyplot as plt

def plot_irl_summary(irl_results, fontsize=15):

    fig = plt.figure()
    
    gd_iterations = np.arange(len(irl_results['opt_val_list']))

    # Plot the optimal forward value as a function of iterations.
    ax = fig.add_subplot(231)
    ax.plot(gd_iterations, irl_results['opt_val_list'], linewidth=3)
    ax.set_xlabel('Gradient Descent Iterations', fontsize=fontsize)
    ax.set_ylabel('Forward problem value', fontsize=fontsize)
    ax.grid()

    ax = fig.add_subplot(232)
    ax.plot(gd_iterations, irl_results['irl_objective_list'], linewidth=3)
    ax.set_xlabel('Gradient Descent Iterations', fontsize=fontsize)
    ax.set_ylabel('IRL objective value', fontsize=fontsize)
    ax.grid()

    ax = fig.add_subplot(233)
    ax.plot(gd_iterations, [np.linalg.norm(diff.flatten()) for diff in irl_results['grad_list']], linewidth=3)
    ax.set_xlabel('Gradient Descent Iterations', fontsize=fontsize)
    ax.set_ylabel('Feature count mismatch', fontsize=fontsize)
    ax.grid()

    ax = fig.add_subplot(234)
    ax.plot(gd_iterations, [irl_results['feature_counts'][0,1] for i in range(len(gd_iterations))], linewidth=3, linestyle='--')
    ax.plot(gd_iterations, [irl_results['state_act_vars_list'][i][(0,1)] for i in range(len(gd_iterations))], linewidth=3)
    ax.grid()

    ax = fig.add_subplot(235)
    ax.plot(gd_iterations, [irl_results['feature_counts'][3,6] for i in range(len(gd_iterations))], linewidth=3, linestyle='--')
    ax.plot(gd_iterations, [irl_results['state_act_vars_list'][i][(3,6)] for i in range(len(gd_iterations))], linewidth=3)
    ax.grid()

    ax = fig.add_subplot(236)
    ax.plot(gd_iterations, [irl_results['feature_counts'][4,7] for i in range(len(gd_iterations))], linewidth=3, linestyle='--')
    ax.plot(gd_iterations, [irl_results['state_act_vars_list'][i][(4,7)] for i in range(len(gd_iterations))], linewidth=3)
    ax.grid()

    plt.show()
