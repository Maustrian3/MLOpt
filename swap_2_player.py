import copy
import random

import numpy as np

from neighborhood import Neighborhood
from problem_instance import ProblemInstance


# Swap two players with the same
class Swap2Player(Neighborhood):
    """Implements a neighborhood structure that swaps two players from different groups."""

    def __init__(self):
        self.week_idx_a = None
        self.week_idx_b = None
        self.group_idx_a = None
        self.group_idx_b = None
        self.player_idx_b = None
        self.player_idx_a = None
        self.old_group_a = None
        self.old_group_b = None
        self.new_conflicts = None
        super().__init__()

    def generate_random_neighborhood_move(self, problem_inst: ProblemInstance, schedule, conflicts):
        # Pick two random weeks
        self.week_idx_a = random.randint(0, problem_inst.num_weeks - 1)
        self.week_idx_b = random.randint(0, problem_inst.num_weeks - 1)

        # Pick two random groups in each week
        self.group_idx_a = random.randint(0, problem_inst.num_groups - 1)
        self.group_idx_b = random.randint(0, problem_inst.num_groups - 1)

        # Pick random players in each group
        self.player_idx_a = random.randint(0, problem_inst.groupsize - 1)
        self.player_idx_b = random.randint(0, problem_inst.groupsize - 1)

        self.old_group_a = schedule[self.week_idx_a][self.group_idx_a]
        self.old_group_b = schedule[self.week_idx_b][self.group_idx_b]
        player_a = self.old_group_a[self.player_idx_a]
        player_b = self.old_group_b[self.player_idx_b]

        # Calculate delta conflicts
        delta = self._calculate_swap_delta(
            conflicts,
            self.old_group_a, self.old_group_b,
            player_a, player_b
        )

        print("Move Infos: \n"
              f"Delta: {delta}\n"
              f"WeekA: {self.week_idx_a} WeekB: {self.week_idx_b}\n"
              f"GroupA: {self.group_idx_a} GroupB: {self.group_idx_b}\n"
              f"PlayerIDXA: {self.player_idx_a} PlayerIDXB: {self.player_idx_b}")

        return delta

    def apply_neighborhood_move(self, schedule, conflicts):
        old_schedule = copy.deepcopy(schedule) # TODO Remove, only for printing used

        player_a = self.old_group_a[self.player_idx_a]
        player_b = self.old_group_b[self.player_idx_b]

        self._update_conflicts(conflicts)

        # Perform swap
        schedule[self.week_idx_a][self.group_idx_a][self.player_idx_a] = player_b
        schedule[self.week_idx_b][self.group_idx_b][self.player_idx_b] = player_a

        # self._print_neighborhood_move(old_schedule, schedule)

        return self.new_conflicts

    def _calculate_swap_delta(self, conflicts, group_a, group_b, player_a, player_b):
        """Calculate change in violations"""
        delta = 0

        # Process group A (player_a leaves, player_b joins)
        for p in group_a:
            if p == player_a:
                continue

            # player_a is leaving this group
            count_a = conflicts[player_a][p]
            # Violation removed if count goes from 2 to 1
            if count_a == 2:
                delta -= 1

            # player_b is joining this group
            count_b = conflicts[player_b][p]
            # Violation created if count goes from 1 to 2
            if count_b == 1:
                delta += 1

        # Process group B (player_b leaves, player_a joins)
        for p in group_b:
            if p == player_b:
                continue

            # player_b is leaving this group
            count_b = conflicts[player_b][p]
            # Violation removed if count goes from 2 to 1
            if count_b == 2:
                delta -= 1

            # player_a is joining this group
            count_a = conflicts[player_a][p]
            # Violation created if count goes from 1 to 2
            if count_a == 1:
                delta += 1

        return delta

    def _update_conflicts(self, conflicts):
        player_a = self.old_group_a[self.player_idx_a]
        player_b = self.old_group_b[self.player_idx_b]

        # Update conflicts for group A
        for p in self.old_group_a:
            if p != player_a:
                conflicts[player_a][p] -= 1
                conflicts[p][player_a] -= 1

                conflicts[player_b][p] += 1
                conflicts[p][player_b] += 1

        # Update conflicts for group B
        for p in self.old_group_b:
            if p != player_b:
                conflicts[player_b][p] -= 1
                conflicts[p][player_b] -= 1

                conflicts[player_a][p] += 1
                conflicts[p][player_a] += 1

    def _print_neighborhood_move(self, old_schedule, new_schedule):
        output = ""
        for week_idx, week in enumerate(old_schedule):
            output += f"Week {week_idx}:\n"

            for group_idx, group in enumerate(week):
                output += f"  Group {group_idx}: ["

                # Schedule
                for player_idx, player in enumerate(group):
                    if week_idx == self.week_idx_a and group_idx == self.group_idx_a and player_idx == self.player_idx_a:
                        output += "a!"
                    if week_idx == self.week_idx_b and group_idx == self.group_idx_b and player_idx == self.player_idx_b:
                        output += "b!"
                    output += f"{player}, "

                output += "] --> ["

                new_group = new_schedule[week_idx][group_idx]
                # Potential new schedule
                for player_idx, player in enumerate(new_group):
                    if week_idx == self.week_idx_a and group_idx == self.group_idx_a and player_idx == self.player_idx_a:
                        output += "b!"
                    if week_idx == self.week_idx_b and group_idx == self.group_idx_b and player_idx == self.player_idx_b:
                        output += "a!"
                    output += f"{player}, "
                output += "]\n"

        print(output)
    
    
