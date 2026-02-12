class ALNSArm:
    """
    Represents a single 'arm' or destroy operator in the Multi-Armed Bandit.
    """
    def __init__(self, name: str, destroy_operator):
        self.name = name
        self.destroy_operator = destroy_operator

        # Weight determines selection probability (roulette wheel)
        self.weight = 1.0
        
        # Statistics
        self.times_used = 0
        self.total_reward = 0.0

    def update(self, reward: float):
        """
        Update the arm's weight based on reward.
        Using Exponential Moving Average (EMA) for adaptation.
        """
        self.times_used += 1
        self.total_reward += reward
        
        # Learning rate (rho)
        rho = 0.1
        self.weight = max(0.01, (1 - rho) * self.weight + rho * reward)