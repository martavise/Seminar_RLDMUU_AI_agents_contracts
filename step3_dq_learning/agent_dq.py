# agent.py
import numpy as np

class AgentDQ:
    """
    Adaption to deep Q-learning.
    The agent's network Q_phi estimates the truncated optimal Q-values Q_bar_phi(s, a),
    representing its payoffs minus the expected immediate payment.
    """
    def __init__(self, mdp, epsilon=0.1):
        self.mdp        = mdp
        self.n_states   = mdp.n_states
        self.n_actions  = mdp.n_actions
        self.n_outcomes = mdp.n_outcomes
        self.gamma      = mdp.gamma
        self.epsilon    = epsilon

    def _expected_payment(self, s, a, b):
        """E_o~O(s,a)[b(o)] — b is tuple or np.array"""
        return sum(self.mdp.P_outcome[s, a, o] * b[o]
                   for o in range(self.n_outcomes))

    def _Q_full(self, s, b, a, q_bar_phi):
        """
        q_bar_phi is estimated from the agents network
        Q*((s,b), a) = E[b(o)] + Q_bar_phi(a)
        Q_bar_phi: 1D array shape [n_actions], Q_phi(s).detach() from agent network
        """
        return self._expected_payment(s, a, b) + q_bar_phi[a]

    def act(self, s, b, q_bar_phi):
        """
        Line 4, Algorithm 2: epsilon-greedy action selection.
        b       : contract tuple from principal
        Q_bar_phi: 1D array shape [n_actions], Q_phi(s).detach() from agent network
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        Q_vals = np.array([self._Q_full(s, b, a, q_bar_phi)
                           for a in range(self.n_actions)])
        return int(np.argmax(Q_vals))