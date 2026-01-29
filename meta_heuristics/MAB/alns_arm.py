class ALNSArm:
    def __init__(self, name: str, operator):
        self.name = name
        self.destroy = operator

        self.weight = 1.0
        self.times_used = 0
        self.total_reward = 0.0