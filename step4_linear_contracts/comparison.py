import sys
import os

_here      = os.path.dirname(os.path.abspath(__file__))
_algo_dir  = os.path.join(_here, '..', 'algorithm_comparison')
_qlearn_dir = os.path.join(_algo_dir, 'qlearn')

sys.path.insert(0, _algo_dir)
sys.path.insert(0, _qlearn_dir)
sys.path.insert(0, _here)

import numpy as np
from principal_agent_mdp import PrincipalAgentMDP
from agent_qlearn import AgentQLearn
from principal_qlearn import PrincipalQLearn
from principal_qlinear import PrincipalLinear


def expected_utility(mdp, r_p, s, a_p, b):
    return sum(
        mdp.P_outcome[s, a_p, o] * (r_p[s, o] - b[o])
        for o in range(mdp.n_outcomes)
    )


def run_one_mdp(n_outcomes, n_episodes=3000):
    # random MDP
    mdp     = PrincipalAgentMDP(n_outcomes=n_outcomes)
    cost_aL = np.random.uniform(0, 1)

    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            mdp.P_outcome[s, a, :] = np.random.dirichlet(3 * np.ones(n_outcomes))

    r_p = np.random.uniform(0, 2, size=(mdp.n_states, n_outcomes))

    def R_agent_rand(s, a, b, o, c=cost_aL):
        return (-c if a == 0 else 0.0) + b[o]

    def R_principal_rand(s, b, o, rp=r_p):
        return rp[s, o] - b[o]

    mdp.R_agent     = R_agent_rand
    mdp.R_principal = R_principal_rand

    # train once with Q-learning
    agent     = AgentQLearn(mdp, alpha=0.1, epsilon=0.1)
    principal = PrincipalQLearn(mdp, r_p, alpha=0.1, epsilon=0.1)

    for _ in range(n_episodes):
        s = mdp.s0
        while not mdp.is_terminal(s):
            Q_bar  = agent.get_Q_bar()
            a_p, b = principal.induce_action(s, Q_bar)
            a      = agent.act(s, b)
            o      = mdp.sample_outcome(s, a)
            s2     = mdp.T(s, o)
            agent.update(s, a, o, s2, tuple(0.0 for _ in range(n_outcomes)))
            principal.update(s, a_p, b, o, s2)
            s = s2

    #  compare LP vs linear from the same trained Q_bar
    Q_bar          = agent.get_Q_bar()
    lp_principal   = PrincipalQLearn(mdp, r_p, alpha=0.1, epsilon=0.0)
    lin_principal  = PrincipalLinear(mdp, r_p, alpha=0.1, epsilon=0.0)
    lp_principal.q = principal.q.copy()

    total_lp = total_linear = 0.0
    for s in range(mdp.n_states):
        if mdp.is_terminal(s):
            continue
        a_p      = int(np.argmax(principal.q[s]))
        b_lp     = lp_principal.find_best_contract(s, a_p, Q_bar)
        b_linear = lin_principal.find_best_contract(s, a_p, Q_bar)
        total_lp     += expected_utility(mdp, r_p, s, a_p, b_lp)
        total_linear += expected_utility(mdp, r_p, s, a_p, b_linear)

    gap = total_lp / total_linear if total_linear != 0 else float('inf')
    return total_lp, total_linear, gap


def run_comparison(n_outcomes_list=(2, 3, 4), n_mdp=100, n_episodes=3000, seed=0):
    np.random.seed(seed)
    results = {}

    for n_outcomes in n_outcomes_list:
        print(f"\n=== {n_outcomes} outcomes ===")
        lp_utils, linear_utils, gap_ratios = [], [], []

        for i in range(n_mdp):
            print(f"  MDP {i + 1}/{n_mdp}", end='\r')
            lp, linear, gap = run_one_mdp(n_outcomes, n_episodes)
            lp_utils.append(lp)
            linear_utils.append(linear)
            gap_ratios.append(gap)

        results[n_outcomes] = {
            'lp_mean':     float(np.mean(lp_utils)),
            'lp_std':      float(np.std(lp_utils)),
            'linear_mean': float(np.mean(linear_utils)),
            'linear_std':  float(np.std(linear_utils)),
            'gap_mean':    float(np.mean(gap_ratios)),
            'gap_std':     float(np.std(gap_ratios)),
        }

    print("\nDone!")
    return results


def print_results(results):
    print(f"{'Outcomes':<10} {'LP mean':<12} {'Linear mean':<14} {'Gap mean':<10} {'Bound'}")
    print("-" * 55)
    for n, r in results.items():
        print(f"{n:<10} {r['lp_mean']:<12.4f} {r['linear_mean']:<14.4f} "
              f"{r['gap_mean']:<10.4f} <= {n}")
