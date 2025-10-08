
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden_size = 64

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh()
        )

        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim)
        )

        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        shared = self.shared(x)
        logits = self.policy(shared)
        value = self.value(shared)
        return logits, value
