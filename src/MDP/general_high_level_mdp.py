import numpy as np
from gurobipy import *

class HLMDP(object):
    """
    Class representing the MDP model of the high-level decision making process.
    """

    def __init__(self, S, A, s_i, s_g, s_fail, controller_list, successor_map):
        """
        Inputs
        ------
        S : numpy array
            State space of the high-level MDP.
        A : numpy array
            Action space of the high-level MDP.
        s_i : int
            Integer representation of the initial state in the high-level MDP.
        s_g : int
            Integer representation of the goal state in the high-level MDP.
        s_fail : int
            Integer representation of the abstract high-level failure state.
        controller_list : list
            List of MinigridController objects (the sub-systems being used as 
            components of the overall RL system).
        successor_map : dict
            Dictionary mapping high-level state-action pairs to the next
            high-level state. 
        """     
        self.controller_list = controller_list

        self.state_list = []
        self.S = S
        self.A = A
        self.s_i = s_i
        self.s_g = s_g
        self.s_fail = s_fail

        self.successor = successor_map

        self.N_S = len(self.S) # Number of states in the high-level MDP
        self.N_A = len(self.A) # Number of actions in the high-level MDP

        self.avail_actions = {}
        self._construct_avail_actions()
        self.avail_states = {}
        self._construct_avail_states()

        self.P = np.zeros((self.N_S, self.N_A, self.N_S), dtype=np.float)
        self._construct_transition_function()

        # Using the successor map, construct a predecessor map.
        self.predecessors = {}
        self._construct_predecessor_map()

    def update_transition_function(self):
        """
        Re-construct the transition function to reflect any changes in the empirical 
        measurements of how likely each controller is to succeed.
        """
        self.P = np.zeros((self.N_S, self.N_A, self.N_S), dtype=np.float)
        self._construct_transition_function()

    # def _construct_state_space(self):
    #     for controller_ind in range(len(self.controller_list)):
    #         controller = self.controller_list[controller_ind]
    #         controller_init_states = controller.get_init_states()
    #         controller_final_states = controller.get_final_states()
    #         if controller_init_states not in self.state_list:
    #             self.state_list.append(controller_init_states)
    #         if controller_final_states not in self.state_list:
    #             self.state_list.append(controller_final_states)

    #     self.state_list.append(-1) # Append another state representing the absorbing "task failed" state
    #     self.S = np.arange(len(self.state_list))
        
    #     self.s_i = self.state_list.index(self.init_states)
    #     self.s_g = self.state_list.index(self.goal_states)
    #     self.s_fail = self.state_list.index(-1)

    def _construct_avail_actions(self):
        for s in self.S:
            self.avail_actions[s] = []
            
        for s in self.S:
            for a in range(self.N_A):
                if (s,a) in self.successor.keys():
                    self.avail_actions[s].append(a)

        # for controller_ind in range(len(self.controller_list)):
        #     controller = self.controller_list[controller_ind]
        #     controller_init_states = controller.get_init_states()
        #     init_s = self.state_list.index(controller_init_states)
        #     self.avail_actions[init_s].append(controller_ind)

    #TODO make sure we don't have duplicate states in this list.
    def _construct_avail_states(self):
        for a in self.A:
            self.avail_states[a]=[]

        for s in self.S:
            avail_actions = self.avail_actions[s]
            for action in avail_actions:
                self.avail_states[action].append(s)

    def _construct_transition_function(self):
        for s in self.S:
            for action in self.avail_actions[s]:
                success_prob = self.controller_list[action].get_success_prob()
                next_s = self.successor[(s, action)]

                self.P[s, action, next_s] = success_prob
                self.P[s, action, self.s_fail] = 1 - success_prob

    # def _construct_successor_map(self):
    #     for s in self.S:
    #         avail_actions = self.avail_actions[s]
    #         for action in avail_actions:
    #             controller_next_states = self.controller_list[action].get_final_states()
    #             next_s = self.state_list.index(controller_next_states)

    #             self.successor[(s, action)] = next_s

    def _construct_predecessor_map(self):
        for s in self.S:
            self.predecessors[s] = []
            for sp in self.S:
                avail_actions = self.avail_actions[sp]
                for action in avail_actions:
                    if self.successor[(sp, action)] == s:
                        self.predecessors[s].append((sp, action))

    def solve_feasible_policy(self, prob_threshold):
        """
        If a meta-policy exists that reaches the goal state from the target 
        state with probability above the specified threshold, return it.

        Inputs
        ------
        prob_threshold : float
            Value between 0 and 1 that represents the desired probability of 
            reaching the goal.

        Outputs
        -------
        policy : numpy (N_S, N_A) array
            Array representing the solution policy. If there is no feasible 
            solution, an array of -1 is returned.
        feasible_flag : bool
            Flag indicating whether or not a feasible solution was found.
        """
        self.update_transition_function()

        if prob_threshold>1 or prob_threshold<0:
            raise RuntimeError("prob threshold is not a probability")

        #initialize gurobi model
        linear_model = Model("abs_mdp_linear")

        #dictionary for state action occupancy
        state_act_vars=dict()

        avail_actions = self.avail_actions.copy()

        #dummy action for goal state
        avail_actions[self.s_g]=[0]

        #create occupancy measures, probability variables and reward variables
        for s in self.S:
            for a in avail_actions[s]:
                state_act_vars[s,a]=linear_model.addVar(lb=0, 
                                        name="state_act_"+str(s)+"_"+str(a))

        #gurobi updates model
        linear_model.update()

        #MDP bellman or occupancy constraints for each state
        for s in self.S:
            cons=0
            #add outgoing occupancy for available actions
            for a in avail_actions[s]:
                cons+=state_act_vars[s,a]

            #add ingoing occupancy for predecessor state actions
            for s_bar, a_bar in self.predecessors[s]:
                #this if clause ensures that you dont double count reaching goal and failure
                if not s_bar == self.s_g and not s_bar == self.s_fail:
                    cons -= state_act_vars[s_bar, a_bar] * self.P[s_bar, a_bar, s]
            #initial state occupancy
            if s == self.s_i:
                cons=cons-1

            #sets occupancy constraints
            linear_model.addConstr(cons==0)

        # prob threshold constraint
        for s in self.S:
            if s == self.s_g:
                linear_model.addConstr(state_act_vars[s,0] >= prob_threshold)

        # set up the objective
        obj=0

        #set the objective, solve the problem
        linear_model.setObjective(obj,GRB.MINIMIZE)
        linear_model.optimize()

        if linear_model.SolCount == 0:
            feasible_flag = False
        else:
            feasible_flag = True

        if feasible_flag:
            # Construct the policy from the occupancy variables
            policy = np.zeros((self.N_S, self.N_A), dtype=np.float)
            for s in self.S:
                if len(self.avail_actions[s]) == 0:
                    policy[s, :] = -1 # If no actions are available, return garbage value
                else:
                    occupancy_state = np.sum([state_act_vars[s,a].x for a in self.avail_actions[s]])
                    # If the state has no occupancy measure under the solution, set the policy to 
                    # be uniform over available actions
                    if occupancy_state == 0.0:
                        for a in self.avail_actions[s]:
                            policy[s,a] = 1 / len(self.avail_actions[s])
                    if occupancy_state > 0.0:
                        for a in self.avail_actions[s]:
                            policy[s, a] = state_act_vars[s,a].x / occupancy_state
        else:
            policy = -1 * np.ones((self.N_S, self.N_A), dtype=np.float)

        return policy, feasible_flag

    def solve_max_reach_prob_policy(self):
        """
        Find the meta-policy that maximizes probability of reaching the goal state.

        Outputs
        -------
        policy : numpy (N_S, N_A) array
            Array representing the solution policy. If there is no feasible solution, an array of
            -1 is returned.
        reach_prob : float
            The probability of reaching the goal state under the policy.
        feasible_flag : bool
            Flag indicating whether or not a feasible solution was found.
        """
        self.update_transition_function()

        #initialize gurobi model
        linear_model = Model("abs_mdp_linear")

        #dictionary for state action occupancy
        state_act_vars=dict()

        avail_actions = self.avail_actions.copy()

        #dummy action for goal state
        avail_actions[self.s_g]=[0]

        #create occupancy measures, probability variables and reward variables
        for s in self.S:
            for a in avail_actions[s]:
                state_act_vars[s,a]=linear_model.addVar(lb=0,name="state_act_"+str(s)+"_"+str(a))

        #gurobi updates model
        linear_model.update()

        #MDP bellman or occupancy constraints for each state
        for s in self.S:
            cons=0
            #add outgoing occupancy for available actions
            for a in avail_actions[s]:
                cons+=state_act_vars[s,a]

            #add ingoing occupancy for predecessor state actions
            for s_bar, a_bar in self.predecessors[s]:
                #this if clause ensures that you dont double count reaching goal and failure
                if not s_bar == self.s_g and not s_bar == self.s_fail:
                    cons -= state_act_vars[s_bar, a_bar] * self.P[s_bar, a_bar, s]
            #initial state occupancy
            if s == self.s_i:
                cons=cons-1

            #sets occupancy constraints
            linear_model.addConstr(cons==0)

        # set up the objective
        obj = 0
        obj+= state_act_vars[self.s_g, 0] # Probability of reaching goal state

        #set the objective, solve the problem
        linear_model.setObjective(obj, GRB.MAXIMIZE)
        linear_model.optimize()

        if linear_model.SolCount == 0:
            feasible_flag = False
        else:
            feasible_flag = True

        if feasible_flag:
            # Construct the policy from the occupancy variables
            policy = np.zeros((self.N_S, self.N_A), dtype=np.float)
            for s in self.S:
                if len(self.avail_actions[s]) == 0:
                    policy[s, :] = -1 # If no actions are available, return garbage value
                else:
                    occupancy_state = np.sum([state_act_vars[s,a].x for a in self.avail_actions[s]])
                    # If the state has no occupancy measure under the solution, set the policy to 
                    # be uniform over available actions
                    if occupancy_state == 0.0:
                        for a in self.avail_actions[s]:
                            policy[s,a] = 1 / len(self.avail_actions[s])
                    if occupancy_state > 0.0:
                        for a in self.avail_actions[s]:
                            policy[s, a] = state_act_vars[s,a].x / occupancy_state
        else:
            policy = -1 * np.ones((self.N_S, self.N_A), dtype=np.float)

        reach_prob = state_act_vars[self.s_g, 0].x

        return policy, reach_prob, feasible_flag

    def solve_low_level_requirements_action(self, prob_threshold, max_timesteps_per_component=None):
        """
        Find new transition probabilities guaranteeing that a feasible meta-policy exists.

        Inputs
        ------
        prob_threshold : float
            The required probability of reaching the target set in the HLM.
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
        reach_prob : float
            The HLM predicted probability of reaching the target set under the solution
            meta-policy and solution transition probabilities in the HLM.
        feasibility_flag : bool
            Flag indicating the feasibility of the bilinear program being solved.
        """
        if prob_threshold > 1 or prob_threshold < 0:
            raise RuntimeError("prob threshold is not a probability")

        # initialize gurobi model
        bilinear_model = Model("abs_mdp_bilinear")

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
        self.avail_actions[self.s_g] = [0]

        # create occupancy measures, probability variables and reward variables
        #for s in self.S:
        for s in self.S:
            for a in self.avail_actions[s]:
                state_act_vars[s,a] = bilinear_model.addVar(lb=0,name="state_act_"+str(s)+"_"+str(a))

        for a in self.A:
            MDP_prob_vars[a] = bilinear_model.addVar(lb=0, ub=1, name="mdp_prob_"  + str(a))
            slack_prob_vars[a] = bilinear_model.addVar(lb=0, ub=1, name="slack_" + str(a))

            MDP_prob_diff_maximizers[a] = bilinear_model.addVar(lb=0, name='mdp_prob_difference_maximizer_'  + str(a))

        # #epigraph variable for max probability constraint
        # prob_maximizer = bilinear_model.addVar(lb=0, name="prob_maximizer")

        # gurobi updates model
        bilinear_model.update()

        # MDP bellman or occupancy constraints for each state
        for s in self.S:
            cons = 0
            # add outgoing occupancy for available actions

            for a in self.avail_actions[s]:
                cons += state_act_vars[s, a]

            # add ingoing occupancy for predecessor state actions
            for s_bar, a_bar in self.predecessors[s]:
                # this if clause ensures that you dont double count reaching goal and failure
                if not s_bar == self.s_g and not s_bar == self.s_fail:
                    cons -= state_act_vars[s_bar, a_bar] * MDP_prob_vars[a_bar]
            # initial state occupancy
            if s == self.s_i:
                cons = cons - 1

            # sets occupancy constraints
            bilinear_model.addConstr(cons == 0)

        # prob threshold constraint
        for s in self.S:
            if s == self.s_g:
                bilinear_model.addConstr(state_act_vars[s, 0] >= prob_threshold)
        print("opt")

        # For each low-level component, add constraints corresponding to
        # the existing performance.
        #for s in self.S:
        for a in self.A:
            existing_success_prob = np.copy(self.controller_list[a].get_success_prob())
            assert (existing_success_prob >= 0 and existing_success_prob <= 1)
            bilinear_model.addConstr(MDP_prob_vars[a] >= existing_success_prob)

        # If one of the components exceeds the maximum allowable training steps, upper bound its success probability.
        if max_timesteps_per_component:
            for a in self.A:
                if self.controller_list[a].data['total_training_steps'] >= max_timesteps_per_component:
                    existing_success_prob = np.copy(self.controller_list[a].get_success_prob())
                    assert (existing_success_prob >= 0 and existing_success_prob <= 1)
                    print('Controller {}, max success prob: {}'.format(a, existing_success_prob))
                    bilinear_model.addConstr(MDP_prob_vars[a] <= existing_success_prob + slack_prob_vars[a])

        # set up the objective
        obj = 0

        slack_cons = 1e3
        # # Minimize the sum of success probability lower bounds

        for a in self.A:
            obj += MDP_prob_diff_maximizers[ a]
            obj += slack_cons * slack_prob_vars[a]

        # Minimize the sum of differences between probability objective and empirical achieved probabilities
        for a in self.A:
            bilinear_model.addConstr(
                MDP_prob_diff_maximizers[a] >= MDP_prob_vars[a] - self.controller_list[a].get_success_prob())

        # set the objective, solve the problem
        bilinear_model.setObjective(obj, GRB.MINIMIZE)
        bilinear_model.optimize()

        if bilinear_model.SolCount == 0:
            feasible_flag = False
        else:
            feasible_flag = True

        for a in self.A:
            if slack_prob_vars[ a].x > 1e-6:
                print("required slack value {} at action: {} ".format(slack_prob_vars[a].x, a))

        if feasible_flag:
            # Update the requirements for the individual components
            required_success_probs = {}
            for a in self.A:
                if a not in required_success_probs.keys():
                    required_success_probs[a] = []
                    required_success_probs[a].append(np.copy(MDP_prob_vars[a].x))
            for a in self.A:
                self.controller_list[a].data['required_success_prob'] = np.max(required_success_probs[a])

            # Create a list of the required success probabilities of each of the components
            required_success_probs = [MDP_prob_vars[a].x for a in self.A]

            # Save the probability of reaching the goal state under the solution
            reach_prob = state_act_vars[self.s_g, 0].x

            # Construct the policy from the occupancy variables
            policy = np.zeros((self.N_S, self.N_A), dtype=np.float)
            for s in self.S:
                if len(self.avail_actions[s]) == 0:
                    policy[s, :] = -1  # If no actions are available, return garbage value
                else:
                    occupancy_state = np.sum([state_act_vars[s, a].x for a in self.avail_actions[s]])
                    # If the state has no occupancy measure under the solution, set the policy to
                    # be uniform over available actions
                    if occupancy_state == 0.0:
                        for a in self.avail_actions[s]:
                            policy[s, a] = 1 / len(self.avail_actions[s])
                    if occupancy_state > 0.0:
                        for a in self.avail_actions[s]:
                            policy[s, a] = state_act_vars[s, a].x / occupancy_state
        else:
            policy = -1 * np.ones((self.N_S, self.N_A), dtype=np.float)
            required_success_probs = [[-1 for a in self.avail_actions[s]] for s in self.S]
            reach_prob = -1

        # Remove dummy action from goal state
        self.avail_actions[self.s_g].remove(0)

        return policy, required_success_probs, reach_prob, feasible_flag