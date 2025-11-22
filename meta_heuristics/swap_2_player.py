from __future__ import annotations

import random
from typing import List

import numpy as np

from neighborhood import Neighborhood
from problem_instance import ProblemInstance


# Swap two players with the same
class Swap2Player(Neighborhood):
    """Implements a neighborhood structure that swaps two players from different groups.

    In this implementation, the move is restricted to a single week:
    two player positions inside that week are swapped. This preserves the
    multiset of players appearing in each week and avoids duplicated
    players within a week, assuming the initial schedule is feasible.

    The change in the number of violating pairs (delta) is computed by
    simulating the swap on a copy of the schedule and recomputing the
    violations from scratch. This keeps the delta and the reported
    violation count consistent with the actual schedule.
    """

    def __init__(self):
        # indices describing the last proposed move
        self.week_idx_a: int | None = None
        self.week_idx_b: int | None = None
        self.group_idx_a: int | None = None
        self.group_idx_b: int | None = None
        self.player_idx_a: int | None = None
        self.player_idx_b: int | None = None

        # groups before the swap (references into the schedule)
        self.old_group_a: List[int] | None = None
        self.old_group_b: List[int] | None = None
        super().__init__()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _violation_count_from_schedule(schedule, num_players: int) -> int:
        """Compute the number of violating pairs directly from the schedule.

        A violating pair is a pair of players that has been grouped
        together more than once across all weeks.
        """
        conflicts = [[0] * num_players for _ in range(num_players)]

        for week in schedule:
            for group in week:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        a = group[i]
                        b = group[j]
                        conflicts[a][b] += 1
                        conflicts[b][a] += 1

        violations = 0
        for i in range(num_players):
            for j in range(i + 1, num_players):
                if conflicts[i][j] > 1:
                    violations += 1
        return violations

    @staticmethod
    def _copy_schedule(schedule):
        """Create a deep copy of the schedule (weeks -> groups -> players)."""
        return [[list(group) for group in week] for week in schedule]

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------
    def generate_random_neighborhood_move(self, problem_inst: ProblemInstance, schedule, conflicts) -> int:
        """Pick a random swap move and return its objective delta.

        The delta is defined as:
            delta = violations_after_swap - violations_before_swap
        and is computed by simulating the swap on a copy of the schedule.
        """
        num_weeks = problem_inst.num_weeks
        num_groups = problem_inst.num_groups
        group_size = problem_inst.groupsize

        # 1) choose a week
        week_idx = random.randint(0, num_weeks - 1)

        # 2) choose two distinct positions (group, index) inside that week
        group_idx_a = random.randint(0, num_groups - 1)
        player_idx_a = random.randint(0, group_size - 1)

        while True:
            group_idx_b = random.randint(0, num_groups - 1)
            player_idx_b = random.randint(0, group_size - 1)
            # ensure we don't pick the exact same position
            if group_idx_b != group_idx_a or player_idx_b != player_idx_a:
                break

        # store move information
        self.week_idx_a = week_idx
        self.week_idx_b = week_idx
        self.group_idx_a = group_idx_a
        self.group_idx_b = group_idx_b
        self.player_idx_a = player_idx_a
        self.player_idx_b = player_idx_b

        self.old_group_a = schedule[week_idx][group_idx_a]
        self.old_group_b = schedule[week_idx][group_idx_b]

        # compute delta by simulating the swap on a copy
        before = self._violation_count_from_schedule(schedule, problem_inst.num_players)

        schedule_copy = self._copy_schedule(schedule)
        a = schedule_copy[week_idx][group_idx_a][player_idx_a]
        b = schedule_copy[week_idx][group_idx_b][player_idx_b]
        schedule_copy[week_idx][group_idx_a][player_idx_a] = b
        schedule_copy[week_idx][group_idx_b][player_idx_b] = a

        after = self._violation_count_from_schedule(schedule_copy, problem_inst.num_players)
        delta = after - before



        return delta

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    def apply_neighborhood_move(self, schedule, conflicts) -> None:
        """Perform the previously generated swap on the schedule.

        The conflict matrix is not updated here; it is recomputed from
        the schedule by SolutionInstance after each accepted move.
        """
        if self.week_idx_a is None or self.group_idx_a is None or self.group_idx_b is None:
            raise RuntimeError("No move has been generated before apply_neighborhood_move was called.")

        if self.player_idx_a is None or self.player_idx_b is None:
            raise RuntimeError("Player indices not set before applying neighborhood move.")

        week = self.week_idx_a
        ga, gb = self.group_idx_a, self.group_idx_b
        ia, ib = self.player_idx_a, self.player_idx_b

        a = schedule[week][ga][ia]
        b = schedule[week][gb][ib]

        schedule[week][ga][ia] = b
        schedule[week][gb][ib] = a
