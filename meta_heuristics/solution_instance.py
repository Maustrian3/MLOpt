from __future__ import annotations

import random
from enum import Enum

import numpy as np

from meta_heuristics.adaptive_large_neighborhood_search import ALNS
from meta_heuristics.problem_instance import ProblemInstance


class InitMethodEnum(Enum):
    RANDOM = 'random'
    GREEDY = 'greedy'
    ROUND_ROBIN = 'round_robin'


class SolutionInstance:
    """Represents a complete solution/schedule and manages the conflict tracking.

    The only ground truth for violations is the schedule itself.
    The conflict matrix is always rebuilt from the schedule when needed,
    so that the reported violation count is consistent with the actual
    pairings in the schedule.
    """

    def __init__(self, problem_instance: ProblemInstance, initial_solution_method):
        self.problem_inst = problem_instance
        self.schedule: list[list[list[int]]] = []
        self.conflicts: np.ndarray | None = None
        self.violation_count: int = 0
        self.alns = ALNS(problem_instance)

        if initial_solution_method is None:
            initial_solution_method = SolutionInstance.generate_random_solution

        # Build initial schedule
        initial_solution_method(self)
        # Ensure conflicts/violations are consistent with the schedule
        self.rebuild_conflicts_and_violations()

    # ------------------------------------------------------------------
    # Objective handling
    # ------------------------------------------------------------------
    @staticmethod
    def is_better_obj(old_obj: int, new_obj: int) -> bool:
        """Return True if the delta new_obj is strictly improving.

        In simulated annealing we pass old_obj=0 and new_obj=delta, so
        this returns True exactly when delta < 0.
        """
        return new_obj < old_obj

    def calc_objective(self) -> int:
        """Return the current objective value (sum of conflicts).

        This is not used by the SA driver for acceptance, but can be
        handy for debugging. It is defined as the sum of the conflict
        matrix entries.
        """
        if self.conflicts is None:
            self.rebuild_conflicts_and_violations()
        return int(self.conflicts.sum())

    # ------------------------------------------------------------------
    # Initial solution
    # ------------------------------------------------------------------
    def generate_random_solution(self) -> None:
        """Generate a random feasible schedule.

        For each week the players are shuffled, then split into equally
        sized groups. One player will sit out each week when the number
        of players is not divisible by groups * groupsize.
        """
        players = list(range(self.problem_inst.num_players))
        num_groups = self.problem_inst.num_groups
        group_size = self.problem_inst.groupsize

        self.schedule = []

        for _week in range(self.problem_inst.num_weeks):
            random.shuffle(players)
            week_schedule: list[list[int]] = []
            for g in range(num_groups):
                start = g * group_size
                end = start + group_size
                group = players[start:end]
                week_schedule.append(group)
            self.schedule.append(week_schedule)

    def generate_greedy_solution(self) -> None:
        """Placeholder for a greedy construction heuristic (not implemented)."""
        raise NotImplementedError("Greedy initialisation is not implemented yet.")

    # ------------------------------------------------------------------
    # Conflict / violation computation from schedule
    # ------------------------------------------------------------------
    def rebuild_conflicts_and_violations(self) -> int:
        """Rebuild the conflict matrix and violation count from the schedule.

        This scans all weeks and groups and counts how many times each
        pair of players has been grouped together.
        """
        n = self.problem_inst.num_players
        conflicts = np.zeros((n, n), dtype=np.int16)

        for week in self.schedule:
            for group in week:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        a = group[i]
                        b = group[j]
                        conflicts[a, b] += 1
                        conflicts[b, a] += 1

        # Number of violating pairs = number of (i,j) with conflicts[i,j] > 1
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                if conflicts[i, j] > 1:
                    violations += 1

        self.conflicts = conflicts
        self.violation_count = violations
        return violations

    def count_violations(self) -> int:
        """Public wrapper to recompute and return the current violation count."""
        return self.rebuild_conflicts_and_violations()

    def repair(self):
        self.schedule = self.alns.repair(self.schedule)

    def destroy_weeks_2(self):
        self.schedule = self.alns.destroy_weeks(self.schedule, 2)

    def destroy_groups_03(self):
        self.schedule = self.alns.destroy_groups(self.schedule, 0.3)

    def destroy_player_pairs_2(self):
        self.schedule = self.alns.destroy_conflicting_pairs(self.schedule, self.conflicts, 2)

    def copy(self) -> SolutionInstance:
        new = SolutionInstance(self.problem_inst, None)
        new.schedule = [[group.copy() for group in week] for week in self.schedule]
        new.conflicts = None
        new.violation_count = self.violation_count
        return new

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        output = f"Violation Count: {self.violation_count}\n"
        for week_idx, week in enumerate(self.schedule):
            output += f"Week {week_idx}:\n"
            for group_idx, group in enumerate(week):
                output += f"  Group {group_idx}: {group}\n"
        return output


if __name__ == '__main__':
    problem_instance = ProblemInstance(3, 10, 4)
    solution_instance = SolutionInstance(problem_instance, SolutionInstance.generate_random_solution)
    print(solution_instance)
