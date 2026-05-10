
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pulp import (
    LpProblem, LpMaximize, LpVariable,
    lpSum, PULP_CBC_CMD, value
)
   
   
# CHANGES for DQ-learning 

# 1. Principals network estimates the contractual optimal Q-values q_theta(s,a)
#       it represents its payoffs when optimally incentivizing the agent to take action a in state s

# 2. grid step removed => continuous b_values in defined range [0,3] (LP)

# 3. agent_Q_bar is now a 1D array as an output from de QNetwork from the agent 

class PrincipalDQ:
    """
    The principal learns two things:
      - which action it wants the agent to take (a_p)
      - what is the cheapest contract that makes the agent do it


    Each step:
      1. induce_action(s, agent_Q_bar): choose action + contract
      2. environment step: observe outcome o and next state s'
    """

    def __init__(self, mdp, r_p, epsilon):
        self.mdp = mdp
        self.n_actions = mdp.n_actions
        self.n_states = mdp.n_states
        self.n_outcomes = mdp.n_outcomes
        self.r_p = r_p
        self.gamma = mdp.gamma   # discount factor
        self.epsilon = epsilon   # exploration rate
        self._contract_cache = {}   # cache to make learning lighter

    def induce_action(self, state, q_theta):
        """
        Input: q_theta contractual optimal Q-values estimated by the principals network
        Line 4 Algorithm 2: Select an action to recommend a_p with q_theta via e-greedy

        """ 
        state_tensor = torch.zeros(self.n_states, dtype=torch.float32).to(device)
        state_tensor[state] = 1.0
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)   # explore
        return int(torch.argmax(q_theta(state_tensor)).item())  # exploit
   
    def find_best_contract(self, s, a_p, agent_Q_bar, lowBound=0, upBound=3):
        """
        Line 11, Algorithm 2: Find optimal contracts b*(s, a_p) by solving LP (17)

        Objective:
            max_{b in B} E_{o~O(s,a_p)}[-b(o)]

        Constraint:
            E[b(o)|a_p] + Q_bar(a_p) >= E[b(o)|a] + Q_bar(a)  for all a != a_p

        Input:
            s           : current state
            a_p         : recommended action (from Line 4 or Line 10)
            agent_Q_bar : 1D array shape [n_actions], Q_phi(s).detach() from agent network
        """
        # cache key
        key = (s, a_p, tuple(agent_Q_bar))
        if key in self._contract_cache:
            return self._contract_cache[key]
        
        mdp = self.mdp

        prob = LpProblem("Best_Contract", LpMaximize)

        b = [LpVariable(f"b_{o}", lowBound=lowBound, upBound=upBound)
             for o in range(self.n_outcomes)]

        prob += lpSum(-mdp.P_outcome[s, a_p, o] * b[o]
                     for o in range(self.n_outcomes))

        for a in range(self.n_actions):
            if a == a_p:
                continue
            prob += (
                lpSum(mdp.P_outcome[s, a_p, o] * b[o]
                      for o in range(self.n_outcomes))
                + agent_Q_bar[a_p]
                >=
                lpSum(mdp.P_outcome[s, a, o] * b[o]
                      for o in range(self.n_outcomes))
                + agent_Q_bar[a]
            )

        prob.solve(PULP_CBC_CMD(msg=0))

        if prob.status == 1:
            b_cont = np.array([value(b[o]) for o in range(self.n_outcomes)])
            # changed to continuous values 
            result = tuple(float(b_cont[i]) for i in range(self.n_outcomes))
        else:
            # if no solution found, offer zero payment
            result = tuple(0.0 for _ in range(self.n_outcomes))
        
        #caching
        self._contract_cache[key] = result
        return result
        
    def reset(self):
        self._contract_cache = {}

