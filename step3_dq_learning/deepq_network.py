import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QNetwork(nn.Module):
    """
    Shared Q-Network architecture for both principal (θ) and agent (ϕ).
    
    Input:  state s
    Output: Q-values for all actions a ∈ A  →  shape [n_actions]

    Architecture (Paper S.35):
        - 2 hidden fully connected layers, each 256 neurons + ReLU
    """
    def __init__(self, n_states, n_actions, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

class ReplayBuffer:
    """
    Sample the transitions used to update the networks 
    In the principal agent MDP: (states, actions, r_agent, r_principal, outcomes, done, next_states)
    done is a binary variable indicating termination (Algo 2) 
    """
    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    # this method samples transitions and returns tensors of each type registered in the environment step
    def sample(self, sample_size):
        sample = random.sample(self.memory, sample_size)
        states = []
        actions = []
        r_agent = []
        r_principal = []
        outcomes = []
        dones = []
        next_states = []
     
        for x in sample:
            states.append(x[0])
            actions.append(x[1])
            r_agent.append(x[2])
            r_principal.append(x[3])
            outcomes.append(x[4])
            dones.append(x[5])
            next_states.append(x[6])
           
        states = torch.tensor(states).to(device)
        actions = torch.tensor(actions).to(device)
        r_agent = torch.tensor(r_agent, dtype=torch.int).to(device)
        r_principal = torch.tensor(r_principal, dtype=torch.float32).to(device)
        outcomes = torch.tensor(outcomes).to(device)
        dones = torch.tensor(dones, dtype=torch.int).to(device)
        next_states = torch.tensor(next_states).to(device)
        
        return states, actions, r_agent, r_principal, outcomes, dones, next_states
    
    # add transition to the buffer
    def append(self, item):
        self.memory.append(item)

    def __len__(self):
        return len(self.memory)