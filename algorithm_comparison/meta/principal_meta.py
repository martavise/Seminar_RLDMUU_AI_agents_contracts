import numpy as np

class PrincipalMeta:

    def __init__(self, mdp, r_p, b_grid_step=0.1):

        self.mdp = mdp
        self.n_states = mdp.n_states
        self.n_outcomes = mdp.n_outcomes
        self.gamma = mdp.gamma
        self.r_p = r_p

        # discretized contract space (still an approximation)
        self.b_values = np.round(np.arange(0, 1.01, b_grid_step), 3)
        self.contracts = np.array(
            np.meshgrid(*[self.b_values]*self.n_outcomes)
        ).T.reshape(-1, self.n_outcomes)

        self.rho_star = None
        self.V = None

    # ---------------------------------------------------
    # Algorithm 1: Principal Bellman Operator
    # ---------------------------------------------------
    def solve(self, agent, tol=1e-6, max_iter=500):

        mdp = self.mdp
        nS = self.n_states

        # =================================================
        # STEP 1: Agent best-response oracle π*(s,b)
        # =================================================
        pi_star = agent.pi_star

        # =================================================
        # STEP 2: INITIALIZE PRINCIPAL VALUE FUNCTION
        # =================================================
        V = np.zeros(nS)
        rho = {s: np.zeros(self.n_outcomes) for s in range(nS)}

        # =================================================
        # STEP 3: VALUE ITERATION (ALG 1 CORE OPERATOR)
        # =================================================
        for it in range(max_iter):

            V_new = np.zeros(nS)
            rho_new = {}

            delta = 0.0

            for s in range(nS):

                best_val = -np.inf
                best_b = None

                # ------------------------------------------------
                # principal chooses contract b
                # ------------------------------------------------
                for b in self.contracts:

                    a = pi_star(s, b)

                    q_val = 0.0

                    # ------------------------------------------------
                    # Bellman expectation over outcomes
                    # ------------------------------------------------
                    for o in range(self.n_outcomes):

                        p = mdp.P_outcome[s, a, o]
                        s_next = mdp.T(s, o)

                        q_val += p * (
                            self.r_p[s, o]
                            - b[o]
                            + self.gamma * V[s_next]
                        )

                    # ------------------------------------------------
                    # contract selection (principal best response)
                    # ------------------------------------------------
                    if q_val > best_val:
                        best_val = q_val
                        best_b = b

                V_new[s] = best_val
                rho_new[s] = best_b

                delta = max(delta, abs(V_new[s] - V[s]))

            V = V_new
            rho = rho_new

            # ------------------------------------------------
            # convergence of Bellman operator
            # ------------------------------------------------
            if delta < tol:
                print(f"[Principal] converged in {it} iterations (Δ={delta:.6e})")
                break

        self.V = V
        self.rho_star = rho

        return V, rho