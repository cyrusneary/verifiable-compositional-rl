from tkinter import font
import numpy as np
import matplotlib.pyplot as plt

def plot_irl_summary(irl_results, fontsize=12):

    fig = plt.figure()
    
    gd_iterations = np.arange(len(irl_results['opt_val_list']))

    # Plot the optimal forward value as a function of iterations.
    ax = fig.add_subplot(131)
    ax.plot(gd_iterations, irl_results['opt_val_list'], linewidth=3)
    ax.set_xlabel('Gradient Descent Iterations', fontsize=fontsize)
    ax.set_ylabel('Forward problem value', fontsize=fontsize)
    ax.grid()

    ax = fig.add_subplot(132)
    ax.plot(gd_iterations, irl_results['irl_objective_list'], linewidth=3)
    ax.set_xlabel('Gradient Descent Iterations', fontsize=fontsize)
    ax.set_ylabel('IRL objective value', fontsize=fontsize)
    ax.grid()

    ax = fig.add_subplot(133)
    ax.plot(gd_iterations, [np.linalg.norm(diff.flatten()) for diff in irl_results['grad_list']], linewidth=3)
    ax.set_xlabel('Gradient Descent Iterations', fontsize=fontsize)
    ax.set_ylabel('Feature count mismatch', fontsize=fontsize)
    ax.grid()

    # ax = fig.add_subplot(234)
    # ax.plot(gd_iterations, [irl_results['feature_counts'][0,1] for i in range(len(gd_iterations))], linewidth=3, linestyle='--')
    # ax.plot(gd_iterations, [irl_results['state_act_vars_list'][i][(0,1)] for i in range(len(gd_iterations))], linewidth=3)
    # ax.grid()

    # ax = fig.add_subplot(235)
    # ax.plot(gd_iterations, [irl_results['feature_counts'][3,6] for i in range(len(gd_iterations))], linewidth=3, linestyle='--')
    # ax.plot(gd_iterations, [irl_results['state_act_vars_list'][i][(3,6)] for i in range(len(gd_iterations))], linewidth=3)
    # ax.grid()

    # ax = fig.add_subplot(236)
    # ax.plot(gd_iterations, [irl_results['feature_counts'][4,7] for i in range(len(gd_iterations))], linewidth=3, linestyle='--')
    # ax.plot(gd_iterations, [irl_results['state_act_vars_list'][i][(4,7)] for i in range(len(gd_iterations))], linewidth=3)
    # ax.grid()

    plt.show()

    fig = plt.figure()
    
    ax = fig.add_subplot(221)
    ax.plot(gd_iterations, [irl_results['feature_counts'][0,1] for i in range(len(gd_iterations))], linewidth=3, linestyle='--', label='Expert Feature Counts')
    ax.plot(gd_iterations, [irl_results['state_act_vars_list'][i][(0,1)] for i in range(len(gd_iterations))], linewidth=3, label='IRL Feature Counts')
    ax.set_xlabel('Gradient Descent Iterations', fontsize=fontsize)
    ax.set_ylabel('Discounted subsystem occupancy measure', fontsize=fontsize)
    ax.set_title('Subsystem 1', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.grid()

    ax = fig.add_subplot(222)
    ax.plot(gd_iterations, [irl_results['feature_counts'][1,2] for i in range(len(gd_iterations))], linewidth=3, linestyle='--')
    ax.plot(gd_iterations, [irl_results['state_act_vars_list'][i][(1,2)] for i in range(len(gd_iterations))], linewidth=3)
    ax.set_xlabel('Gradient Descent Iterations', fontsize=fontsize)
    ax.set_ylabel('Discounted subsystem occupancy measure', fontsize=fontsize)
    ax.set_title('Subsystem 2', fontsize=fontsize)
    ax.grid()

    ax = fig.add_subplot(223)
    ax.plot(gd_iterations, [irl_results['feature_counts'][3,6] for i in range(len(gd_iterations))], linewidth=3, linestyle='--')
    ax.plot(gd_iterations, [irl_results['state_act_vars_list'][i][(3,6)] for i in range(len(gd_iterations))], linewidth=3)
    ax.set_xlabel('Gradient Descent Iterations', fontsize=fontsize)
    ax.set_ylabel('Discounted subsystem occupancy measure', fontsize=fontsize)
    ax.set_title('Subsystem 6', fontsize=fontsize)
    ax.grid()

    ax = fig.add_subplot(224)
    ax.plot(gd_iterations, [irl_results['feature_counts'][4,7] for i in range(len(gd_iterations))], linewidth=3, linestyle='--')
    ax.plot(gd_iterations, [irl_results['state_act_vars_list'][i][(4,7)] for i in range(len(gd_iterations))], linewidth=3)
    ax.set_xlabel('Gradient Descent Iterations', fontsize=fontsize)
    ax.set_ylabel('Discounted subsystem occupancy measure', fontsize=fontsize)
    ax.set_title('Subsystem 7', fontsize=fontsize)
    ax.grid()

    plt.show()