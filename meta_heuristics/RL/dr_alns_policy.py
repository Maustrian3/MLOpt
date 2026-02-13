import torch.nn as nn

class DRALNSPolicy(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Policy head:
        # Outputs a logit for each destroy operator (action). After softmax, this gives a probability distribution over actions.
        self.policy_head = nn.Linear(64, num_actions)

        # Value head:
        # Predicts the state value, i.e., expected cumulative reward from this state.
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        x = self.shared(state)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

