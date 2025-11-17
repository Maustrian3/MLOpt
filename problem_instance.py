class ProblemInstance:
    """Defines the problem parameters and constraints."""
    def __init__(self, num_groups, num_players, num_weeks):
        self.num_groups = num_groups
        self.num_players = num_players
        self.num_weeks = num_weeks
        self.groupsize = int(self.num_players/self.num_groups)
