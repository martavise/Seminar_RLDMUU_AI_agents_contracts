import numpy as np
## This a discrete MDP with a finite number of states and actions
class PrincipalAgentMDP:
    """
    Hidden-action Principal-Agent MDP — Figure 1, Section 2.2
    M = (S, s0, A, B, O, O, R, Rp, T, gamma)
    PRINCIPAL: offers contract b, observes outcome, cannot see action
    AGENT: chooses action, observes state AND contract
    """
    def __init__(self, gamma):
        self.n_states = 3  # s0=0, sL=1, sR=2 S = {s0, sL, sR})
        self.n_actions = 2  # 2 actions: left, right
        self.n_outcomes = 2 #L=0, R=1
        self.s0 = 0  # s0 initial state
        self.gamma = gamma  # γ
        # P_outcomes[s, a, o] = probability of outcome o given state s and action a
        # same probabilities in every state
        #aL (0): L w.p. 0.9, R w.p. 0.1
        #aR (1): L w.p. 0.1, R w.p. 0.9
        self.P_outcome = np.zeros([self.n_states, self.n_actions, self.n_outcomes])
        for s in range(self.n_states):
            self.P_outcome[s,0] = [0.9, 0.1] #aL
            self.P_outcome[s,1] = [0.1, 0.9] #aR

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


if __name__ == "__main__":
    mdp = PrincipalAgentMDP()
    print("States:", mdp.n_states)
    print("Actions:", mdp.n_actions)
    print("Outcomes:", mdp.n_outcomes)
    o = mdp.sample_outcome(state=0, action=0)
    s_next = mdp.T(state=0, outcome=o)
    print(f"Outcome: {'L' if o==0 else 'R'}, Next state: {['s0','sL','sR'][s_next]}")
