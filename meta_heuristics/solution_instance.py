from __future__ import annotations

import random
from enum import Enum

import numpy as np

from meta_heuristics.neighborhood import Neighborhood
from meta_heuristics.problem_instance import ProblemInstance
from meta_heuristics.swap_2_player import Swap2Player


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

    def __init__(self, problem_inst: ProblemInstance, initial_solution_method, neighborhood: Neighborhood):
        self.problem_instance = problem_inst
        self.schedule: list[list[list[int]]] = []
        # conflict matrix: conflicts[i][j] = number of times i and j were in the same group
        self.conflicts: np.ndarray | None = None
        self.neighborhood = neighborhood
        self.violation_count: int = 0

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
        players = list(range(self.problem_instance.num_players))
        num_groups = self.problem_instance.num_groups
        group_size = self.problem_instance.groupsize

        self.schedule = []

        for _week in range(self.problem_instance.num_weeks):
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
        n = self.problem_instance.num_players
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

    # ------------------------------------------------------------------
    # Neighborhood integration
    # ------------------------------------------------------------------
    def generate_random_neighborhood_move(self) -> int:
        """Ask the neighborhood to propose a move and return its delta.

        The neighborhood is passed the schedule (and the conflict matrix,
        which it may ignore) and must return the change in violations
        if that move were applied.
        """
        if self.conflicts is None:
            self.rebuild_conflicts_and_violations()
        delta = self.neighborhood.generate_random_neighborhood_move(
            self.problem_instance,
            self.schedule,
            self.conflicts,
        )
        return delta

    def apply_neighborhood_move(self, delta: int) -> None:
        """Apply the previously generated move and recompute violations.

        The delta argument is not used to update the objective directly;
        instead the conflicts and violation count are rebuilt from
        the schedule to keep them truthful.
        """
        self.neighborhood.apply_neighborhood_move(self.schedule, self.conflicts)
        # After applying the move, recompute the true conflicts/violations
        self.rebuild_conflicts_and_violations()

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
    neighborhood = Swap2Player()
    solution_instance = SolutionInstance(problem_instance, SolutionInstance.generate_random_solution, neighborhood)
    print(solution_instance)
    delta = solution_instance.generate_random_neighborhood_move()
    solution_instance.apply_neighborhood_move(delta)
    print(solution_instance.violation_count)
    print(solution_instance.count_violations())
