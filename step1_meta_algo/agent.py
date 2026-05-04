import numpy as np

class Agent:
    """
    Agent in a Principal-Agent MDP.
    Solves the inner optimization of the meta-algorithm:
    given principal's policy rho, find best-responding pi*.
    """
    def __init__(self, mdp):
        self.mdp       = mdp
        self.n_states  = mdp.n_states
        self.n_actions = mdp.n_actions
        self.n_outcomes = mdp.n_outcomes
        self.gamma     = mdp.gamma
  

        # these are set after solve() is called
        self.V       = None   # V_agent[s]
        self.Q_agent = None   # Q^pi((s,b), a | rho)
        self.pi_star = None   # pi*(s, b) -> action

    def solve(self, rho, tol=1e-10, max_iter=1000):
        """
        Fix rho: s -> np.array([b(L), b(R)]) and solve agent's MDP
        via value iteration.

        Sets self.V, self.Q_agent, self.pi_star
        """
        mdp = self.mdp
        nS  = self.n_states
        nA  = self.n_actions
        V   = np.zeros(nS)

        # value iteration
        for _ in range(max_iter):
            V_new = np.zeros(nS)
            for s in range(nS):
                b = rho[s]
                Q = np.zeros(nA)
                for a in range(nA):
                    for o in range(mdp.n_outcomes):
                        p      = mdp.P_outcome[s, a, o]
                        s_next = mdp.T(s, o)
                        Q[a]  += p * (mdp.R_agent(s, a, b, o) + self.gamma * V[s_next])
                V_new[s] = np.max(Q)

            if np.max(np.abs(V - V_new)) < tol:
                break
            V = V_new

        self.V = V

        # Q^pi((s,b), a | rho) for arbitrary b at current state
        # future states still use rho(s') — captured in V
        def Q_agent(s, b, a):
            q = 0.0
            for o in range(mdp.n_outcomes):
                p      = mdp.P_outcome[s, a, o]
                s_next = mdp.T(s, o)
                q     += p * (mdp.R_agent(s, a, b, o) + self.gamma * V[s_next])
            return q

        # pi*(s, b) = argmax_a Q^pi((s,b), a)
        # ties broken in favor of principal (principal Q injected later if needed)
        def pi_star(s, b):
            Q_vals  = np.array([Q_agent(s, b, a) for a in range(nA)])
            max_val = np.max(Q_vals)
            tied    = np.where(np.abs(Q_vals - max_val) < 1e-9)[0]
            return tied[0]

        self.Q_agent = Q_agent
        self.pi_star = pi_star

        return V, Q_agent, pi_star
    