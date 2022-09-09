import numpy as np
import cvxpy as cp
from tqdm import tqdm
import gurobipy as gb

def solve_max_reward_perfect_subsystems(mdp,
                                        reward_vec : np.ndarray):
    #initialize gurobi model
    linear_model = gb.Model("abs_mdp_linear")

    #dictionary for state action occupancy
    state_act_vars=dict()

    avail_actions = mdp.avail_actions.copy()

    #dummy action for goal state
    avail_actions[mdp.s_g]=[0]

    #create occupancy measures, probability variables and reward variables
    for s in mdp.S:
        for a in avail_actions[s]:
            state_act_vars[s,a] = linear_model.addVar(lb=0, 
                                        name="state_act_"+str(s)+"_"+str(a))

    #gurobi updates model
    linear_model.update()

    #MDP bellman or occupancy constraints for each state
    for s in mdp.S:
        cons=0
        #add outgoing occupancy for available actions
        for a in avail_actions[s]:
            cons+=state_act_vars[s,a]

        #add ingoing occupancy for predecessor state actions
        for s_bar, a_bar in mdp.predecessors[s]:
            #this if clause ensures that you dont double count reaching goal and failure
            if not s_bar == mdp.s_g and not s_bar == mdp.s_fail:
                cons -= mdp.discount * state_act_vars[s_bar, a_bar] * 1.0 # optimism
        #initial state occupancy
        if s == mdp.s_i:
            cons=cons-1

        #sets occupancy constraints
        linear_model.addConstr(cons==0)

    obj = 0
    for s in mdp.S:
        for a in mdp.avail_actions[s]:
            obj += reward_vec[s,a] * state_act_vars[s,a]

    #set the objective, solve the problem
    linear_model.setObjective(obj, gb.GRB.MAXIMIZE)
    linear_model.optimize()

    if linear_model.SolCount == 0:
        feasible_flag = False
    else:
        feasible_flag = True

    if feasible_flag:
        # Construct the policy from the occupancy variables
        policy = np.zeros((mdp.N_S, mdp.N_A), dtype=np.float)
        for s in mdp.S:
            if len(mdp.avail_actions[s]) == 0:
                policy[s, :] = -1 # If no actions are available, return garbage value
            else:
                occupancy_state = np.sum([state_act_vars[s,a].x for a in mdp.avail_actions[s]])
                # If the state has no occupancy measure under the solution, set the policy to 
                # be uniform over available actions
                if occupancy_state == 0.0:
                    for a in mdp.avail_actions[s]:
                        policy[s,a] = 1 / len(mdp.avail_actions[s])
                if occupancy_state > 0.0:
                    for a in mdp.avail_actions[s]:
                        policy[s, a] = state_act_vars[s,a].x / occupancy_state
    else:
        policy = -1 * np.ones((mdp.N_S, mdp.N_A), dtype=np.float)

    reward_max = 0
    for s in mdp.S:
        for a in mdp.avail_actions[s]:
            reward_max += reward_vec[s,a] * state_act_vars[s,a].x

    return policy, reward_max, feasible_flag            

def solve_max_reward(mdp,
                    reward_vec : np.ndarray):
    #initialize gurobi model
    linear_model = gb.Model("abs_mdp_linear")

    #dictionary for state action occupancy
    state_act_vars=dict()

    avail_actions = mdp.avail_actions.copy()

    #dummy action for goal state
    avail_actions[mdp.s_g]=[0]

    #create occupancy measures, probability variables and reward variables
    for s in mdp.S:
        for a in avail_actions[s]:
            state_act_vars[s,a] = linear_model.addVar(lb=0, 
                                        name="state_act_"+str(s)+"_"+str(a))

    #gurobi updates model
    linear_model.update()

    #MDP bellman or occupancy constraints for each state
    for s in mdp.S:
        cons=0
        #add outgoing occupancy for available actions
        for a in avail_actions[s]:
            cons+=state_act_vars[s,a]

        #add ingoing occupancy for predecessor state actions
        for s_bar, a_bar in mdp.predecessors[s]:
            #this if clause ensures that you dont double count reaching goal and failure
            if not s_bar == mdp.s_g and not s_bar == mdp.s_fail:
                cons -= mdp.discount * state_act_vars[s_bar, a_bar] * mdp.controller_list[a_bar].get_success_prob() # optimism
        #initial state occupancy
        if s == mdp.s_i:
            cons=cons-1

        #sets occupancy constraints
        linear_model.addConstr(cons==0)

    obj = 0
    for s in mdp.S:
        for a in mdp.avail_actions[s]:
            obj += reward_vec[s,a] * state_act_vars[s,a]

    #set the objective, solve the problem
    linear_model.setObjective(obj, gb.GRB.MAXIMIZE)
    linear_model.optimize()

    if linear_model.SolCount == 0:
        feasible_flag = False
    else:
        feasible_flag = True

    if feasible_flag:
        # Construct the policy from the occupancy variables
        policy = np.zeros((mdp.N_S, mdp.N_A), dtype=np.float)
        for s in mdp.S:
            if len(mdp.avail_actions[s]) == 0:
                policy[s, :] = -1 # If no actions are available, return garbage value
            else:
                occupancy_state = np.sum([state_act_vars[s,a].x for a in mdp.avail_actions[s]])
                # If the state has no occupancy measure under the solution, set the policy to 
                # be uniform over available actions
                if occupancy_state == 0.0:
                    for a in mdp.avail_actions[s]:
                        policy[s,a] = 1 / len(mdp.avail_actions[s])
                if occupancy_state > 0.0:
                    for a in mdp.avail_actions[s]:
                        policy[s, a] = state_act_vars[s,a].x / occupancy_state
    else:
        policy = -1 * np.ones((mdp.N_S, mdp.N_A), dtype=np.float)

    reward_max = 0
    for s in mdp.S:
        for a in mdp.avail_actions[s]:
            reward_max += reward_vec[s,a] * state_act_vars[s,a].x

    return policy, reward_max, feasible_flag   

def solve_low_level_requirements_action(mdp, 
                                        reward_vec : np.ndarray,
                                        delta : float, 
                                        reward_max : float,
                                        max_timesteps_per_component : int = None):
    """
    Find new transition probabilities guaranteeing that a feasible meta-policy 
    exists with expected reward >= delta * reward_max.

    Inputs
    ------
    reward_vec :
        An array with shape (mdp.N_S, mdp.N_A) that encodes the reward function.
    delta : float
        The required probability of reaching the target set in the HLM.
    reward_max : float
        The maximum achievable reward with perfect subsystems.
    max_timesteps_per_component : int
        Number of training steps (for an individual sub-system) beyond which its current
        estimated performance value should be used as an upper bound on the corresponding
        transition probability in the HLM.

    Outputs
    -------
    policy : numpy array
        The meta-policy satisfying the task specification, under the solution
        transition probabilities in the HLM.
        Returns an array of -1 if no feasible solution exists.
    required_success_probs : list
        List of the solution transition probabilities in the HLM.
        Returns a list of -1 if no feasible solution exists.
    achieved_reward : float
        The HLM predicted expected reward under the solution
        meta-policy and solution transition probabilities in the HLM.
    feasibility_flag : bool
        Flag indicating the feasibility of the bilinear program being solved.
    """
    if delta > 1 or delta < 0:
        raise RuntimeError("delta value should be between 0 and 1")

    # initialize gurobi model
    bilinear_model = gb.Model("abs_mdp_bilinear")

    # activate gurobi nonconvex
    bilinear_model.params.NonConvex = 2

    # dictionary for state action occupancy
    state_act_vars = dict()

    # dictionary for MDP prob variables
    MDP_prob_vars = dict()

    # dictionary for slack variables
    slack_prob_vars = dict()

    # dictionary for epigraph variables used to define objective
    MDP_prob_diff_maximizers = dict()

    # dummy action for goal state
    mdp.avail_actions[mdp.s_g] = [0]

    # create occupancy measures, probability variables and reward variables
    #for s in self.S:
    for s in mdp.S:
        for a in mdp.avail_actions[s]:
            state_act_vars[s,a] = bilinear_model.addVar(lb=0,name="state_act_"+str(s)+"_"+str(a))

    for a in mdp.A:
        MDP_prob_vars[a] = bilinear_model.addVar(lb=0, ub=1, name="mdp_prob_"  + str(a))
        slack_prob_vars[a] = bilinear_model.addVar(lb=0, ub=1, name="slack_" + str(a))

        MDP_prob_diff_maximizers[a] = bilinear_model.addVar(lb=0, name='mdp_prob_difference_maximizer_'  + str(a))

    # gurobi updates model
    bilinear_model.update()

    # MDP bellman or occupancy constraints for each state
    for s in mdp.S:
        cons = 0
        # add outgoing occupancy for available actions

        for a in mdp.avail_actions[s]:
            cons += state_act_vars[s, a]

        # add ingoing occupancy for predecessor state actions
        for s_bar, a_bar in mdp.predecessors[s]:
            # # this if clause ensures that you dont double count reaching goal and failure
            # if not s_bar == mdp.s_g and not s_bar == mdp.s_fail:
            cons -= mdp.discount * state_act_vars[s_bar, a_bar] * MDP_prob_vars[a_bar]
        # initial state occupancy
        if s == mdp.s_i:
            cons = cons - 1

        # sets occupancy constraints
        bilinear_model.addConstr(cons == 0)

    # Expected reward constraint
    rew_sum = 0
    for s in mdp.S:
        for a in mdp.avail_actions[s]:
            rew_sum += reward_vec[s,a] * state_act_vars[s,a]
    bilinear_model.addConstr(rew_sum >= (1-delta) * reward_max)

    print("opt")

    # For each low-level component, add constraints corresponding to
    # the existing performance.
    for a in mdp.A:
        existing_success_prob = np.copy(mdp.controller_list[a].get_success_prob())
        assert (existing_success_prob >= 0 and existing_success_prob <= 1)
        bilinear_model.addConstr(MDP_prob_vars[a] >= existing_success_prob)

    # If one of the components exceeds the maximum allowable training steps, upper bound its success probability.
    if max_timesteps_per_component:
        for a in mdp.A:
            if mdp.controller_list[a].data['total_training_steps'] >= max_timesteps_per_component:
                existing_success_prob = np.copy(mdp.controller_list[a].get_success_prob())
                assert (existing_success_prob >= 0 and existing_success_prob <= 1)
                print('Controller {}, max success prob: {}'.format(a, existing_success_prob))
                bilinear_model.addConstr(MDP_prob_vars[a] <= existing_success_prob + slack_prob_vars[a])

    # set up the objective
    obj = 0

    slack_cons = 1e3
    # # Minimize the sum of success probability lower bounds

    for a in mdp.A:
        obj += MDP_prob_diff_maximizers[a]
        obj += slack_cons * slack_prob_vars[a]

    # Minimize the sum of differences between probability objective and empirical achieved probabilities
    for a in mdp.A:
        bilinear_model.addConstr(
            MDP_prob_diff_maximizers[a] >= MDP_prob_vars[a] - mdp.controller_list[a].get_success_prob())

    # set the objective, solve the problem
    bilinear_model.setObjective(obj, gb.GRB.MINIMIZE)
    bilinear_model.optimize()

    if bilinear_model.SolCount == 0:
        feasible_flag = False
    else:
        feasible_flag = True

    for a in mdp.A:
        if slack_prob_vars[a].x > 1e-6:
            print("required slack value {} at action: {} ".format(slack_prob_vars[a].x, a))

    if feasible_flag:
        # Update the requirements for the individual components
        required_success_probs = {}
        for a in mdp.A:
            if a not in required_success_probs.keys():
                required_success_probs[a] = []
                required_success_probs[a].append(np.copy(MDP_prob_vars[a].x))
        for a in mdp.A:
            mdp.controller_list[a].data['required_success_prob'] = np.max(required_success_probs[a])

        # Create a list of the required success probabilities of each of the components
        required_success_probs = [MDP_prob_vars[a].x for a in mdp.A]

        # Save the probability of reaching the goal state under the solution
        achieved_reward = 0
        for s in mdp.S:
            for a in mdp.avail_actions[s]:
                achieved_reward += reward_vec[s,a] * state_act_vars[s,a].x

        # Construct the policy from the occupancy variables
        policy = np.zeros((mdp.N_S, mdp.N_A), dtype=np.float)
        for s in mdp.S:
            if len(mdp.avail_actions[s]) == 0:
                policy[s, :] = -1  # If no actions are available, return garbage value
            else:
                occupancy_state = np.sum([state_act_vars[s, a].x for a in mdp.avail_actions[s]])
                # If the state has no occupancy measure under the solution, set the policy to
                # be uniform over available actions
                if occupancy_state == 0.0:
                    for a in mdp.avail_actions[s]:
                        policy[s, a] = 1 / len(mdp.avail_actions[s])
                if occupancy_state > 0.0:
                    for a in mdp.avail_actions[s]:
                        policy[s, a] = state_act_vars[s, a].x / occupancy_state
    else:
        policy = -1 * np.ones((mdp.N_S, mdp.N_A), dtype=np.float)
        required_success_probs = [[-1 for a in mdp.avail_actions[s]] for s in mdp.S]
        achieved_reward = -1

    # Remove dummy action from goal state
    mdp.avail_actions[mdp.s_g].remove(0)

    return policy, required_success_probs, achieved_reward, feasible_flag