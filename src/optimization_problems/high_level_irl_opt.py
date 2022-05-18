import numpy as np
import cvxpy as cp
from tqdm import tqdm

def solve_optimistic_irl(mdp, 
                        feature_counts : np.ndarray, 
                        num_iterations : int = 1000, 
                        alpha : float = 0.01,
                        verbose : bool = False):
    """
    Solve an inverse reinforcement learning problem.
    """
    irl_results = {
        'feature_counts' : feature_counts,
        'init_theta' : np.zeros(feature_counts.shape),
        # np.copy(feature_counts), # As a starting guess, just use the provided empirical discounted feature counts
        'theta_list' : [],
        'opt_val_list' : [],
        'state_act_vars_list' : [],
        'state_vars_list' : [],
        'grad_list' : [],
        'irl_objective_list' : [],
    }

    # Begin by constructing the optimization problem
    opt_problem = construct_optimistic_irl_forward_pass(mdp)

    irl_results['theta_list'].append(irl_results['init_theta'])

    if verbose: print('Solving inverse RL problem.')
    for i in tqdm(range(num_iterations)):
        
        theta = irl_results['theta_list'][-1]

        # solve the forward problem
        sol_val = solve_optimistic_forward_problem(opt_problem, theta)

        # save the results of the forward problem
        irl_results['opt_val_list'].append(sol_val)
        irl_results['state_act_vars_list'].append(extract_opt_var_values(opt_problem['vars']['state_act_vars']))
        irl_results['state_vars_list'].append(extract_opt_var_values(opt_problem['vars']['state_vars']))

        irl_results['irl_objective_list'].append(sol_val - np.sum(np.multiply(theta, feature_counts)))

        # Calculate gradient w.r.t. theta and take a step.
        grad = state_act_feature_count_difference(feature_counts, 
                                        opt_problem['vars']['state_act_vars'])
        irl_results['grad_list'].append(grad)

        new_theta = theta - alpha * grad

        # Project theta so all elements are positive
        # new_theta[np.where(new_theta < 0)] = 0

        irl_results['theta_list'].append(new_theta)

    return irl_results

def extract_opt_var_values(vars_dict):
    copy = {}
    for key in vars_dict.keys():
        copy[key] = vars_dict[key].value
    return copy

def extract_optimal_policy(vars_dict):
    policy = {}
    for s in vars_dict['state_vars'].keys():
        for (s,a) in vars_dict['state_act_vars'].keys():
            pass

def solve_optimistic_forward_problem(problem : dict,
                                    reward_vec : np.ndarray) -> float:
    """
    Solve the forward pass of the inverse reinforcement learning problem.
    This corresponds to solving a maximum entropy dynamic programming problem 
    with a given reward vector.

    Parameters
    ----------
    problem : 
        A dictionary containing a cvxpy optimization problem (problem['problem']),
        its variables (problem['vars']), and its parameters (problem['params']).
    reward_vec : 
        A (N_S, N_A) numpy array representing the reward vector to use in 
        solving the forward optimization problem.

    Returns
    -------
    The optimal value of the optimization problem, corresponding to the 
    optimal discounted reward.
    """
    assert (problem['params']['reward_vec'].value.shape == reward_vec.shape)
    problem['params']['reward_vec'].value = reward_vec
    return problem['problem'].solve(verbose=False)

def state_act_feature_count_difference(demo_feature_count : np.ndarray, 
                                        opt_state_act_vars : dict) -> np.ndarray:
    """
    Compute the difference between the optimal state-action discounted feature 
    counts obtained from the last optimization solve, and the empirical feature
    counts obtained empirically.

    Parameters
    ----------
    demo_feature_count :
        The empirical features counts measured from demonstration.
        demo_feature_count[s,a] returns the occupancy measure for action "a"
        taken from state "s".
    opt_state_act_vars : 
        The feature counts from the last optimization problem solve.
        opt_state_act_vars[(s,a)].value returns the occupancy measure value
        for action "a" taken from state "s".

    Returns
    -------
    diff :
        An array with shape (N_S, N_A) representing the difference in the 
        feature count values.
    """
    diff = np.zeros(demo_feature_count.shape)
    for (s,a) in opt_state_act_vars.keys():
        diff[s, a] = opt_state_act_vars[(s,a)].value - demo_feature_count[s,a]
    return diff

# def state_feature_count_difference(demo_feature_count, opt_state_vars):
#     for s in range(len(demo_feature_count)):
#         diff[s] = opt_state_vars[s] - demo_feature_count[s]
#     return diff

def construct_optimistic_irl_forward_pass(mdp):
    #dictionary for state occupancy and state action occupancy
    state_vars = dict()
    state_act_vars = dict()

    avail_actions = mdp.avail_actions.copy()

    #create occupancy measures, probability variables and reward variables
    for s in mdp.S:
        state_vars[s] = cp.Variable(name="state_"+str(s), nonneg=True)

        for a in avail_actions[s]:
            state_act_vars[(s, a)] = cp.Variable(name="state_act_"+str(s)+"_"+str(a), 
                                                nonneg=True)

    vars = {
        'state_act_vars' : state_act_vars,
        'state_vars' : state_vars
    }
    
    # Create problem parameters
    reward_vec = cp.Parameter(shape=(mdp.N_S, mdp.N_A), 
                                name='reward_vec',
                                value=np.zeros((mdp.N_S, mdp.N_A)))
    params = {
        'reward_vec' : reward_vec
    }

    ###### define the problem constraints
    cons = []

    #MDP bellman or occupancy constraints for each state
    for s in mdp.S:
        cons_sum=0

        cons_sum += state_vars[s]

        #add ingoing occupancy for predecessor state actions
        for s_bar, a_bar in mdp.predecessors[s]:
            # #this if clause ensures that you dont double count reaching goal and failure
            # if not s_bar == mdp.s_g and not s_bar == mdp.s_fail:
            cons_sum -= mdp.discount * state_act_vars[s_bar, a_bar] * 1.0 #mdp.P[s_bar, a_bar, s]
        #initial state occupancy
        if s == mdp.s_i:
            cons_sum = cons_sum - 1

        #sets occupancy constraints
        cons.append(cons_sum == 0)

    # Define relation between state-action occupancy measures 
    # and state occupancy measures
    for s in mdp.S:
        # Only enforce the following constraint if outgoing actions are available.
        if avail_actions[s]:
            cons_sum = cp.sum([state_act_vars[s,a] for a in avail_actions[s]])
            cons.append(state_vars[s] == cons_sum)

    # set up the objective
    obj_sum = 0

    for s in mdp.S:
        for a in avail_actions[s]:
            obj_sum -= cp.rel_entr(state_act_vars[s,a], state_vars[s])
            obj_sum += reward_vec[s, a] * state_act_vars[s,a]

    obj = cp.Maximize(obj_sum)

    prob = cp.Problem(objective=obj, constraints=cons)

    forward_problem = {
        'problem' : prob,
        'vars' : vars,
        'params' : params
    }

    return forward_problem