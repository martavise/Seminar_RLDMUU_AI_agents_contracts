import numpy as np

class AgentMeta:
    def __init__(self, mdp):
        self.mdp = mdp
        self.n_states = mdp.n_states
        self.n_actions = mdp.n_actions
        self.n_outcomes = mdp.n_outcomes
        self.gamma = mdp.gamma

        self.V = None
        self.Q_agent = None
        self.pi_star = None

    def solve(self, rho, tol=1e-10, max_iter=1000):

        mdp = self.mdp
        nS = self.n_states
        nA = self.n_actions

        V = np.zeros(nS)

        # ----------------------------
        # VALUE ITERATION (AGENT)
        # ----------------------------
        for _ in range(max_iter):
            V_new = np.zeros(nS)

            for s in range(nS):

                b = rho[s]  # contract at state s

                best_val = -np.inf

                for a in range(nA):

                    val = 0.0

                    for o in range(mdp.n_outcomes):

                        p = mdp.P_outcome[s, a, o]
                        s_next = mdp.T(s, o)

                        b_next = rho[s_next]  # IMPORTANT: contract evolves with state

                        val += p * (
                            mdp.R_agent(s, a, b, o)
                            + self.gamma * V[s_next]
                        )

                    best_val = max(best_val, val)

                V_new[s] = best_val

            if np.max(np.abs(V - V_new)) < tol:
                break

            V = V_new

        self.V = V

        # ----------------------------
        # Q FUNCTION (CONSISTENT WITH V)
        # ----------------------------
        def Q_agent(s, b, a):

            q = 0.0

            for o in range(mdp.n_outcomes):

                p = mdp.P_outcome[s, a, o]
                s_next = mdp.T(s, o)

                q += p * (
                    mdp.R_agent(s, a, b, o)
                    + self.gamma * V[s_next]
                )

            return q

        # ----------------------------
        # POLICY (BEST RESPONSE)
        # ----------------------------
        def pi_star(s, b):

            Q_vals = np.array([Q_agent(s, b, a) for a in range(nA)])

            max_val = np.max(Q_vals)
            tied = np.where(np.abs(Q_vals - max_val) < 1e-9)[0]

            return tied[0]

        self.Q_agent = Q_agent
        self.pi_star = pi_star

        return V, Q_agent, pi_star