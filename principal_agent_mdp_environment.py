import numpy as np
## This a discrete MDP with a finite number of states and actions
class PrincipalAgentMDP:
    """
    Hidden-action Principal-Agent MDP — Figure 1, Section 2.2
    M = (S, s0, A, B, O, O, R, Rp, T, gamma)
    PRINCIPAL: offers contract b, observes outcome, cannot see action
    AGENT: chooses action, observes state AND contract
    """
    def __init__(self):
        self.n_states = 3  # s0=0, sL=1, sR=2 S = {s0, sL, sR})
        self.n_actions = 2  # 2 actions: left, right
        self.n_outcomes = 2 #L=0, R=1
        self.s0 = 0  # s0 initial state
        self.gamma = 1.0  # γ
        # B — discrete contract space (supervisor requirement)
        # payments for outcomes L and R, always >= 0 (limited liability)
        self.b_values = np.round(np.arange(0, 2.1, 0.1), 2)
        # P_outcomes[s, a, o] = probability of outcome o given state s and action a
        # same probabilities in every state
        #aL (0): L w.p. 0.9, R w.p. 0.1
        #aR (1): L w.p. 0.1, R w.p. 0.9
        self.P_outcome = np.zeros([self.n_states, self.n_actions, self.n_outcomes])
        for s in range(self.n_states):
            self.P_outcome[s,0] = [0.9, 0.1] #aL
            self.P_outcome[s,1] = [0.1, 0.9] #aR

        # #agent reward r(s,a)
        # r(s, aL) = -4/5 (costly), r(s, aR) = 0 (free)
        self.r_a = np.zeros([self.n_states, self.n_actions])
        for s in range(self.n_states):
            self.r_a[s, 0] = -4 / 5  # aL costs the agent
            self.r_a[s, 1] = 0.0  # aR is free

        # principal reward rp(s, o)
        # rp(s, L) = 14/9, rp(s, R) = 0, same in every state
        self.r_p = np.zeros([self.n_states, self.n_outcomes])
        for s in range(self.n_states):
            self.r_p[s, 0] = 14 / 9  # outcome L rewards principal
            self.r_p[s, 1] = 0.0  # outcome R gives nothing

    def T(self, state, outcome):
        """
        Transition function: outcome → next state
         L(0) → sL(1), R(1) → sR(2)
        Same from any state.
        """
        return outcome + 1  # L(0)→1, R(1)→2

    def sample_outcome(self, state, action):
        """
        Sample an outcome given state and action.
        Corresponds to o ~ O(s, a)
        """
        probs = self.P_outcome[state, action]
        return np.random.choice(self.n_outcomes, p=probs)

    def expected_payment(self, state, action, b_o1, b_o2):
        """
        Expected payment to agent given action and contract (b_o1, b_o2).
        = P(L|s,a) * b_o1 + P(R|s,a) * b_o2
        """
        p_L = self.P_outcome[state, action, 0]
        p_R = self.P_outcome[state, action, 1]
        return p_L * b_o1 + p_R * b_o2

    def get_agent_reward(self, state, action, outcome, b_o1, b_o2):
        """
        Agent's total reward = base reward + payment received.
        R(s, a, b, o) = r(s, a) + b(o)  -- Section 2.2
        """
        base = self.r_a[state, action]
        payment = b_o1 if outcome == 0 else b_o2
        return base + payment

    def get_principal_reward(self, state, outcome, b_o1, b_o2):
        """
        Principal's total reward = outcome value minus payment made.
        Rp(s, b, o) = rp(s, o) - b(o)  -- Section 2.2
        """
        base = self.r_p[state, outcome]
        payment = b_o1 if outcome == 0 else b_o2
        return base - payment


if __name__ == "__main__":
    mdp = PrincipalAgentMDP()
    print("States:", mdp.n_states)
    print("Actions:", mdp.n_actions)
    print("Outcomes:", mdp.n_outcomes)
    print("Contracts:", mdp.b_values)
    o = mdp.sample_outcome(state=0, action=0)
    s_next = mdp.T(state=0, outcome=o)
    print(f"Outcome: {'L' if o==0 else 'R'}, Next state: {['s0','sL','sR'][s_next]}")
