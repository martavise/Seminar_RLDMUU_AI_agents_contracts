import numpy as np


class PrincipalLinear:
    """
    Principal using linear contracts: b[o] = alpha * r_p[o]
    Mirrors principal_qlearn.py exactly, replacing the LP with a
    grid search over alpha in [0, 1].
    """

    def __init__(self, mdp, r_p, alpha, epsilon, b_grid_step=0.1):
        self.mdp        = mdp
        self.n_actions  = mdp.n_actions
        self.n_states   = mdp.n_states
        self.n_outcomes = mdp.n_outcomes
        self.r_p        = r_p       # r_p[s, o]
        self.alpha      = alpha     # Q-learning rate
        self.gamma      = mdp.gamma
        self.epsilon    = epsilon
        self.q          = np.zeros((self.n_states, self.n_actions))

        # E_rp[s, a] = E[r_p(s,o) | s, a] — precomputed once
        self.E_rp = np.array([
            [sum(mdp.P_outcome[s, a, o] * r_p[s, o] for o in range(mdp.n_outcomes))
             for a in range(mdp.n_actions)]
            for s in range(mdp.n_states)
        ])

    def find_best_contract(self, s, a_p, agent_Q_bar):
        """
        Grid search over alpha in [0, 1]
        For each alpha: b[o] = alpha * r_p[s, o]
        Keep the alpha with highest principal utility (1 - alpha) * E_rp[s, a_p]
        that satisfies IC for all competing actions.
        """
        best_alpha   = None
        best_utility = -np.inf

        for alpha in np.linspace(0, 1, 100):
            b = tuple(alpha * self.r_p[s, o] for o in range(self.n_outcomes))

            # check IC: E[b|a_p] + Q_bar[a_p] >= E[b|a] + Q_bar[a] for all a != a_p
            ic_ok = True
            for a in range(self.n_actions):
                if a == a_p:
                    continue
                lhs = (sum(self.mdp.P_outcome[s, a_p, o] * b[o]
                           for o in range(self.n_outcomes))
                       + agent_Q_bar[s, a_p])
                rhs = (sum(self.mdp.P_outcome[s, a, o] * b[o]
                           for o in range(self.n_outcomes))
                       + agent_Q_bar[s, a])
                if lhs < rhs:
                    ic_ok = False
                    break

            if ic_ok:
                utility = (1 - alpha) * self.E_rp[s, a_p]
                if utility > best_utility:
                    best_utility = utility
                    best_alpha   = alpha

        if best_alpha is None:
            return tuple(0.0 for _ in range(self.n_outcomes))

        return tuple(best_alpha * self.r_p[s, o] for o in range(self.n_outcomes))

    def induce_action(self, state, agent_Q_bar):
        if np.random.rand() < self.epsilon:
            best_a_p = np.random.randint(self.n_actions)
        else:
            best_a_p = int(np.argmax(self.q[state, :]))
        best_b = self.find_best_contract(state, best_a_p, agent_Q_bar)
        return best_a_p, best_b

    def update(self, state, a_p, b, o, next_state):
        reward = self.r_p[state, o] - b[o]
        target = reward + self.gamma * np.max(self.q[next_state, :])
        self.q[state, a_p] += self.alpha * (target - self.q[state, a_p])

    def reset(self):
        self.q = np.zeros((self.n_states, self.n_actions))
