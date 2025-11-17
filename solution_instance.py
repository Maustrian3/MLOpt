import random
from enum import Enum

import numpy as np
from Tools.scripts.texi2html import increment

from neighborhood import Neighborhood
from problem_instance import ProblemInstance
from swap_2_player import Swap2Player


class InitMethodEnum(Enum):
    RANDOM = 'random'
    GREEDY = 'greedy'
    ROUND_ROBIN = 'round_robin'

class SolutionInstance:
    """Represents a complete solution/schedule and manages the conflict tracking."""

    def __init__(self, problem_inst: ProblemInstance, initial_solution_method, neighborhood: Neighborhood):
        self.problem_instance = problem_inst
        self.schedule = []
        self.conflicts = np.zeros((self.problem_instance.num_players, self.problem_instance.num_players),
                                  dtype=np.int8)

        self.neighborhood = neighborhood
        self.violation_count = None

        method_map = {
            InitMethodEnum.RANDOM: self.generate_random_solution,
            InitMethodEnum.GREEDY: self.generate_greedy_solution
        }

        if initial_solution_method is None:
            initial_solution_method = SolutionInstance.generate_random_solution
        initial_solution_method(self)

    def is_better_obj(self, old_obj, new_obj):
        return new_obj < old_obj

    def calc_objective(self):
        return self.conflicts.sum()

    def generate_random_solution(self):
        players = list(range(self.problem_instance.num_players))  # [0, 1, 2, ..., num_players-1]
        self.schedule = []
        self.conflicts = np.zeros(
            (self.problem_instance.num_players, self.problem_instance.num_players),
            dtype=np.int8
        )

        for week in range(self.problem_instance.num_weeks):
            random.shuffle(players)

            # Split into groups
            week_schedule = []
            for g in range(self.problem_instance.num_groups):
                start = g * self.problem_instance.groupsize
                end = start + self.problem_instance.groupsize
                group = players[start:end]
                week_schedule.append(group)

                # Update conflict matrix for this group
                self._update_conflicts_for_group(group, inc=1)

            self.schedule.append(week_schedule)

        self.count_violations()



    def generate_greedy_solution(self):
        pass

    def _update_conflicts_for_group(self, group, inc):
        """Add/remove conflicts for all pairs in a group"""
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                player_a = group[i]
                player_b = group[j]
                self.conflicts[player_a][player_b] += inc
                self.conflicts[player_b][player_a] += inc

    def count_violations(self):
        """Count pairs that played together more than once"""
        # Count all entries > 1 and divide by 2 (since matrix is symmetric)
        self.violation_count = np.sum(self.conflicts > 1) // 2
        return self.violation_count

    def generate_random_neighborhood_move(self):
        delta = self.neighborhood.generate_random_neighborhood_move(self.problem_instance, self.schedule, self.conflicts)
        return delta

    def apply_neighborhood_move(self, delta):
        """Apply move and track violations"""
        self.neighborhood.apply_neighborhood_move(self.schedule, self.conflicts)
        self.violation_count += delta #TODO violation count with delta doesnt match with the one calculated with count_violations()
        self.count_violations() # TODO remove if delta eval works

    def __str__(self):
        output = f"Violation Count: {self.violation_count}\n"
        for week_idx, week in enumerate(self.schedule):
            output += f"Week {week_idx}:\n"
            for group_idx, group in enumerate(week):
                output += f"  Group {group_idx}: {group}\n"
        return output


if __name__ == '__main__':
    problem_instance = ProblemInstance(3,10,4)
    neighborhood = Swap2Player()
    solution_instance = SolutionInstance(problem_instance, SolutionInstance.generate_random_solution, neighborhood)
    print(solution_instance)
    delta = solution_instance.generate_random_neighborhood_move()
    solution_instance.apply_neighborhood_move(delta)

    print(solution_instance.violation_count)
    print(solution_instance.count_violations())