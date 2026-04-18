class PrincipalMeta:
    """
    Principal in a Principal-Agent MDP
    Solves outer optimization of the meta-algorithm:
    given the agents policy pi, find best responding rho 
    """ 

    def __init__(self, mdp, r_p, b_grid_step=0.1):
        self.mdp = mdp
        self.n_states = mdp.n_states
        self.n_outcomes = mdp.n_outcomes
        self.gamma = mdp.gamma
        self.r_p = r_p
        self.b_values = np.round(np.arange(0, 1.01, b_grid_step), 3)
        self.contracts = list(product(self.b_values, repeat=self.n_outcomes))
        self.n_contracts = len(self.contracts)

        self.V = None
        self.Q = None
        self.rho_star = None

    def solve(self, pi, tol=1e-4, max_iter=1000, snapshot_every=1):
        """
        Fix pi: solve principal's MDP via value iteration.
        """
        mdp = self.mdp
        V = {s: 0.0 for s in range(self.n_states)}
        snapshots = []

        for it in range(1, max_iter + 1):
            delta = 0.0
            new_V = {}

            for s in range(self.n_states):
                a = pi[s]

                q_values = []
                for b in self.contracts:
                    q = sum(
                        mdp.P_outcome[s, a, o] * (self.r_p[s, o] - b[o] + self.gamma * V[mdp.T[s, o]])
                        for o in range(self.n_outcomes)
                    )
                    q_values.append(q)

                new_V[s] = max(q_values)
                delta = max(delta, abs(new_V[s] - V[s]))

            V = new_V

            rho_star = {}
            for s in range(self.n_states):
                a = pi[s]
                q_values = [
                    sum(
                        mdp.P_outcome[s, a, o] * (self.r_p[s, o] - b[o] + self.gamma * V[mdp.T[s, o]])
                        for o in range(self.n_outcomes)
                    )
                    for b in self.contracts
                ]
                rho_star[s] = self.contracts[int(np.argmax(q_values))]

            if it % snapshot_every == 0 or delta < tol:
                snapshots.append((V.copy(), rho_star.copy(), delta, it))

            if delta < tol:
                print(f"Principal VI converged in {it} iterations (Δ={delta:.6f})")
                break

        self.V = V
        self.rho_star = rho_star

        return snapshots