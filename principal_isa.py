import numpy as np

from pulp import (
    LpProblem, LpMaximize, LpVariable,
    lpSum, PULP_CBC_CMD, value
)


# CHANGES

# 1. __init__: we no longer store the agent's Q-bar at the start. The agent is still learning, so its Q-bar changes every step.

# 2. induce_action:  look at a scoreboard (self.q) that we update as we learn, and pick the action with the highest score.

# 3. Q_star method removed, the scoreboard (self.q) replaces it.
# it gets better over time through experience.

# 4. find_best_contract: now receives the agent's Q-bar as an input each time, instead of using an old stored version.

# 5. b_values: contracts must stay between 0 and 1. the principal can only pay the agent, never charge them.



class Principal:
    """
    The principal learns two things:
      - which action it wants the agent to take (a_p)
      - what is the cheapest contract that makes the agent do it

    Each step:
      1. induce_action(s, agent_Q_bar): choose action + contract
      2. environment step: observe outcome o and next state s'
      3. update(s, a_p, b, o, s'): update the scoreboard from experience
    """

    def __init__(self, mdp, r_p, alpha, epsilon, b_grid_step=0.1):
        self.mdp = mdp
        self.n_actions = mdp.n_actions
        self.n_states = mdp.n_states
        self.n_outcomes = mdp.n_outcomes
        self.r_p = r_p
        self.alpha = alpha       # learning rate
        self.gamma = mdp.gamma   # discount factor
        self.epsilon = epsilon   # exploration rate
        # grid of possible payment values: [0.0, 0.1, 0.2, ..., 1.0]
        self.b_values = np.round(np.arange(0, 1.01, b_grid_step), 3)
        # scoreboard: q[state, action] = how good is it to induce this action here
        self.q = np.zeros((self.n_states, self.n_actions))
        self._contract_cache = {}   # cache to make learning lighter

    def induce_action(self, state, agent_Q_bar):
        """
        Decide which action to push the agent toward, then find the contract.

        With probability epsilon: explore, pick a random action.
        Otherwise: exploit:  pick the action with the best score on the scoreboard.
        Then: use the LP to find the cheapest payment that makes the agent do it.
        """
        if np.random.rand() < self.epsilon:
            best_a_p = np.random.randint(self.n_actions)   # explore
        else:
            best_a_p = int(np.argmax(self.q[state, :]))    # exploit scoreboard

        best_b = self.find_best_contract(state, best_a_p, agent_Q_bar)
        return best_a_p, best_b

    def find_best_contract(self, s, a_p, agent_Q_bar, lowBound=0, upBound=1):
        """
        Use LP to find the cheapest payment b = [b(L), b(R)]
        that makes the agent prefer action a_p over all other actions.

        The constraint says: taking a_p must be at least as good for the agent
        as taking any other action, given the payment offer.
        """
        # cache key
        key = (s, a_p, tuple(agent_Q_bar.flatten()))
        if key in self._contract_cache:
            return self._contract_cache[key]
        
        mdp = self.mdp

        prob = LpProblem("Best_Contract", LpMaximize)

        # one payment variable per outcome: b(L) and b(R)
        b = [LpVariable(f"b_{o}", lowBound=lowBound, upBound=upBound)
             for o in range(self.n_outcomes)]

        # objective: minimise the expected payment (principal pays as little as possible)
        prob += lpSum(-mdp.P_outcome[s, a_p, o] * b[o]
                     for o in range(self.n_outcomes))

        # constraint: a_p must be at least as attractive as every other action
        for a in range(self.n_actions):
            if a == a_p:
                continue
            prob += (
                lpSum(mdp.P_outcome[s, a_p, o] * b[o]
                      for o in range(self.n_outcomes))
                + agent_Q_bar[s, a_p]
                >=
                lpSum(mdp.P_outcome[s, a, o] * b[o]
                      for o in range(self.n_outcomes))
                + agent_Q_bar[s, a]
            )

        prob.solve(PULP_CBC_CMD(msg=0))

        if prob.status == 1:  # solution found
            b_cont = np.array([value(b[o]) for o in range(self.n_outcomes)])
            # round to nearest value on our payment grid
            result  = tuple(
                min(self.b_values, key=lambda x: abs(x - b_cont[i]))
                for i in range(self.n_outcomes)
            )
        else:
            # if no solution found, offer zero payment
            result = tuple(0.0 for _ in range(self.n_outcomes))
        
        #caching
        self._contract_cache[key] = result
        return result

    def update(self, state, a_p, b, o, next_state):
        """
        Update the scoreboard after observing what happened.
        Standard Q-learning: the new score is the reward we got plus
        the best future score, blended with the old score using alpha.
        """
        reward = self.r_p[state, o] - b[o]
        # no future value at terminal states
        future = 0.0 if self.mdp.is_terminal(next_state) else np.max(self.q[next_state, :])
        target = reward + self.gamma * future
        self.q[state, a_p] += self.alpha * (target - self.q[state, a_p])

    def reset(self):
        # wipe the scoreboard back to zero
        self.q = np.zeros((self.n_states, self.n_actions))
        self._contract_cache = {}
