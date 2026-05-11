# agent.py
import numpy as np

class AgentQLearn:
    """
    Q-learning agent using truncated Q-function (eq. 1, Section 4).
    Q*((s,b), a) = E[b(o)] + Q_bar(s, a)
    
    Q_bar[s, a] is learned via Q-learning, independent of current b.
    This is what Principal.find_best_contract expects as agent_Q_bar.
    """
    def __init__(self, mdp, alpha=0.1, epsilon=0.1):
        self.mdp        = mdp
        self.n_states   = mdp.n_states
        self.n_actions  = mdp.n_actions
        self.n_outcomes = mdp.n_outcomes
        self.alpha      = alpha
        self.gamma      = mdp.gamma
        self.epsilon    = epsilon

        # truncated Q-table: Q_bar[s, a]
        # shape (n_states, n_actions) — this is what principal expects
        self.Q_bar = np.zeros((self.n_states, self.n_actions))

    def reset(self):
        self.Q_bar = np.zeros((self.n_states, self.n_actions))

    def _expected_payment(self, s, a, b):
        """E_o~O(s,a)[b(o)] — b is tuple or np.array"""
        return sum(self.mdp.P_outcome[s, a, o] * b[o]
                   for o in range(self.n_outcomes))

    def _Q_full(self, s, b, a):
        """
        Q*((s,b), a) = E[b(o)] + Q_bar(s, a)
        equation (1) from Section 4
        """
        return self._expected_payment(s, a, b) + self.Q_bar[s, a]

    def act(self, s, b):
        """
        Epsilon-greedy action selection.
        b is a tuple (from Principal.induce_action) or np.array.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        Q_vals = np.array([self._Q_full(s, b, a)
                           for a in range(self.n_actions)])
        return int(np.argmax(Q_vals))

    def update(self, s, a, o, s_next, b_next):
        """
        Q-learning update for Q_bar(s, a).

        target = r(s,a) + gamma * max_a'[E[b_next(o')] + Q_bar(s', a')]

        r(s,a) isolated by passing zero contract to mdp.R_agent.
        b_next is tuple from rho[s_next] — compatible with _expected_payment.
        """
        # r(s,a) only — zero contract isolates effort cost from payment
        r_sa = self.mdp.R_agent(s, a, (0.0, 0.0), o)

        # max_a' Q*((s', b_next), a')
        # no future value at terminal states
        if self.mdp.is_terminal(s_next):
            future = 0.0
        else:
            Q_next = np.array([self._Q_full(s_next, b_next, a)
                            for a in range(self.n_actions)])
            future = np.max(Q_next)
        target = r_sa + self.gamma * future
        self.Q_bar[s, a] += self.alpha * (target - self.Q_bar[s, a])



    def get_Q_bar(self):
        """
        Returns Q_bar[s, a] — shape (n_states, n_actions).
        This is what Principal.find_best_contract expects as agent_Q_bar.
        """
        return self.Q_bar.copy()