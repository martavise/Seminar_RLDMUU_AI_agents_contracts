import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'step2_meta_qlearn'))
sys.path.insert(0, _here)

import numpy as np
from principal_agent_mdp import PrincipalAgentMDP
from agent_qlearn import Agent
from principal_qlearn import Principal
from principal_qlinear import PrincipalLinear


def run_comparison(n_episodes=3000, seed=42):
    mdp = PrincipalAgentMDP()
    r_p = np.array([[14 / 9 if o == 0 else 0.0 for o in range(2)] for s in range(3)])

    lr      = 0.1
    epsilon = 0.1

    def run_meta(PrincipalCls):
        np.random.seed(seed)
        agent     = Agent(mdp, alpha=lr, epsilon=epsilon)
        principal = PrincipalCls(mdp, r_p, alpha=lr, epsilon=epsilon)
        utilities = []

        for _ in range(n_episodes):
            s          = mdp.s0
            ep_utility = 0.0

            while not mdp.is_terminal(s):
                Q_bar    = agent.get_Q_bar() # what has the agent learned so far
                a_p, b   = principal.induce_action(s, Q_bar) # find best action
                a        = agent.act(s, b) # choose action
                o        = mdp.sample_outcome(s, a) # observe outcome that the environment returns
                s2       = mdp.T(s, o) # next state
                ep_utility += mdp.R_principal(s, b, o) # record what principal has learned so far
                agent.update(s, a, o, s2, (0.0, 0.0)) # experience from agent
                principal.update(s, a_p, b, o, s2) # experience from principal
                s = s2

            utilities.append(ep_utility)

        return utilities

    lp_utilities     = run_meta(Principal)
    linear_utilities = run_meta(PrincipalLinear)

    lp_final     = float(np.mean(lp_utilities[-50:]))
    linear_final = float(np.mean(linear_utilities[-50:]))
    gap_ratio    = lp_final / linear_final if linear_final != 0 else float('inf')

    return {
        'lp_utilities':     lp_utilities,
        'linear_utilities': linear_utilities,
        'lp_final':         lp_final,
        'linear_final':     linear_final,
        'gap_ratio':        gap_ratio,
    }


def print_results(results):
    print(f"LP principal utility:      {results['lp_final']:.2f}")
    print(f"Linear principal utility:  {results['linear_final']:.2f}")
    print(f"Gap ratio:                 {results['gap_ratio']:.2f}")
    print(f"Theoretical bound (n=2):   2.00")
