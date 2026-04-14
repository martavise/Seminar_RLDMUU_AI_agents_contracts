import numpy as np

class Agent:
    
    '''
    initialisation of MDP with needed parameters and serialisation of contracts
    '''
    def __init__(self, mdp, alpha=0.1, epsilon=0.1, b_grid_step=0.1):
        self.mdp = mdp
        self.n_states = mdp.n_states
        self.n_actions = mdp.n_actions
        self.n_outcomes = mdp.n_outcomes

        self.alpha = alpha
        self.gamma = mdp.gamma
        self.epsilon = epsilon

        self.cost = np.array([0.1, 0.5])  # effort costs

        self.b_values = np.round(np.arange(0, 1.01, b_grid_step), 3)
        self.b_grid = self._build_contract_grid()

        self.q = np.zeros((self.n_states, len(self.b_grid), self.n_actions))


    def _build_contract_grid(self):
        grid = []
        for bL in self.b_values:
            for bR in self.b_values:
                grid.append((bL, bR))
        return grid

    def contract_to_id(self, b):
        b = np.array(b)

        # snap each component to nearest grid value
        snapped = tuple(
            min(self.b_values, key=lambda x: abs(x - b[i]))
            for i in range(len(b))
        )

        return self.b_grid.index(snapped)
    '''
    makes decision of action accordingly to state and contract, policy
    '''
    def act(self, state, b):
        b_id = self.contract_to_id(b)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        return np.argmax(self.q[state, b_id])

    '''
    returns rewards as benefice of contract minus costs of action
    '''
    def compute_reward(self, s, a, o, b):
        return -self.cost[a] + b[o]

    def update(self, s, b, a, o, s_next, rho):
        r_agent = self.compute_reward(s, a, o, b)

        b_id = self.contract_to_id(b)

        b_next = rho(s_next)
        b_next_id = self.contract_to_id(b_next)

        max_next = np.max(self.q[s_next, b_next_id])

        # adapted Q-learning with contracts
        target = r_agent + self.gamma * max_next
        self.q[s, b_id, a] += self.alpha * (target - self.q[s, b_id, a])

    '''
    calculate the value of Q_bar
    '''
    def get_Q_bar(self, rho):
        Q_bar = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            b = rho(s)
            b_id = self.contract_to_id(b)
            Q_bar[s] = self.q[s, b_id]

        return Q_bar
    

    def reset(self):
        self.q = np.zeros_like(self.q)