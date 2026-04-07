class Principal: 
    """
    Principal learns which action a_p to induce and the optimal contract b.
    The principal's MDP is determined by the agents policy.

    Workflow per state s:
        For each a_p in A:
            1. find_best_contract(s, a_p)
               -- Eq. 3: Solve LP for optimal contract b*
               -- max_{b in B} E_{o~O(s,a_p)}[-b(o)]
               -- s.t. E[b(o)|a_p] + Q_bar(s,a_p) >= E[b(o)|a] + Q_bar(s,a) for all a in A

            2. Q_star(s, a_p, b*)
               -- Eq. 12: Evaluate contract b* for action a_p
               -- Q*(s, b | pi) = E_o[r^p(s,o) - b(o) + gamma * max_{a'} q*(s', a')]

            3. q*(s, a_p) = Q*(s, b*)
               -- Eq. 2: Assign value to action a_p

        Select argmax_{a_p} q*(s, a_p) via induce_action()
        Return: (a_p, b) with highest q*

    Learning:
        update(s, a_p, b, o, s') updates q(s, a_p) via Q-learning (sample-based Eq. 12)

    Dependencies:
        - agents_policy (Q_bar): agent's Q-function, learned in inner loop
        - r_p: principal's own reward function r^p(s, o)
        - mdp.P_outcome[s, a, o]: outcome distribution P(o | s, a)
        - mdp.T[s, o]: deterministic state transition s' = T(s, o)
    """

    def __init__(self, mdp, r_p, agents_policy, alpha, epsilon, b_grid_step=0.1): 
        self.mdp = mdp
        self.n_actions = mdp.n_actions 
        self.n_states = mdp.n_states
        self.n_outcomes = mdp.n_outcomes
        self.r_p = r_p # principals reward function 
        self.agents_policy = agents_policy # fixes the MDP for the Principal 
        self.alpha = alpha
        self.gamma = mdp.gamma 
        self.epsilon = epsilon
        self.b_values = np.round(np.arange(0, 2.1, b_grid_step), 2)
        self.q = np.zeros((self.n_states, self.n_actions))
    
    
    def induce_action(self, state):
        """
        Eq. 2: q*(s, a_p) = max_{b: pi*=a_p} Q*(s, b)
        Select argmax_{a_p} q*(s, a_p)

        For each a_p:
            1. find_best_contract  -> b*
            2. Q_star(s, a_p, b*) -> Q*(s, b*)
            3. q*(s, a_p) = Q*    -> assign
        """

        best_a_p = 0
        best_q = - np.inf
        best_b = (0.0,0.0)

        for a_p in range(self.n_actions):
            b = self.find_best_contract(state,a_p)
            Q_val = self.Q_star(state, a_p, b)
            if Q_val > best_q:
                best_q = Q_val
                best_a_p = a_p
                best_b = b 
        return best_a_p, best_b

    def Q_star(self, s, a_p, b): 
        """
        Eq. 12: Q*(s, b | pi) = E_o[r^p(s,o) - b(o) + gamma * max_{a'} q*(s', a')]
        """ 
        val = 0.0
        for o in range(self.n_outcomes):
            p_o = self.mdp.P_outcome[s, a_p, o]
            s_next = self.mdp.T(s, o)
            reward = self.r_p[s, o] - b[o]
            val += p_o * (reward + self.gamma * np.max(self.q[s_next, :]))
        return val 

    def find_best_contract(self, s, a_p, lowBound = 0, upBound = 1): 
        """
        Objective Eq. 3: max_{b in B} E[-b(o) | a_p]
        s.t. E[b(o)|a_p] + Q_bar(s,a_p) >= E[b(o)|a] + Q_bar(s,a)  for all a
        bounds: 0 <= b(o) <= 1
    """
        mdp = self.mdp
        Q_bar = self.agents_policy

        # Problem
        prob = LpProblem(f"Best Contract", LpMaximize)

        # Variables: b(o) in [0, 1]
        b = [LpVariable(f"b_{o}", lowBound=lowBound, upBound=upBound)
            for o in range(self.n_outcomes)]

        # Objective: max E[-b(o) | a_p]
        prob += lpSum(-mdp.P_outcome[s, a_p, o] * b[o]
                  for o in range(self.n_outcomes))

        # Constraints (Eq. 2)
        # E[b(o)|a_p] + Q_bar(s,a_p) >= E[b(o)|a] + Q_bar(s,a)  for all actions a 
        for a in range(self.n_actions):
            if a == a_p:
                continue
            prob += (
                lpSum(mdp.P_outcome[s, a_p, o] * b[o]
                    for o in range(self.n_outcomes))
                + Q_bar[s, a_p]
                >=
                lpSum(mdp.P_outcome[s, a, o] * b[o]
                  for o in range(self.n_outcomes))
                + Q_bar[s, a]
            )

        # Solve
        prob.solve(PULP_CBC_CMD(msg=0))

        if prob.status == 1:  # Optimal
            return tuple(value(b[o]) for o in range(self.n_outcomes))
        else:
            return None       

    def update(self, state, a_p, b, o, next_state):
        """
        Sample-based version of Eq. 12:
        target = r^p(s,o) - b(o) + gamma * max_{a'} q(s', a')
        q(s, a_p) <- q(s, a_p) + alpha * (target - q(s, a_p))
        """
        reward = self.r_p[state, o] - b[o]
        target = reward + self.gamma * np.max(self.q[next_state, :])
        self.q[state, a_p] += self.alpha * (target - self.q[state, a_p])

    def reset(self):
        self.q = np.zeros((self.n_states, self.n_actions))
        

    

    