import random

from constraint_programming.exact_method import solve_social_golfer
from meta_heuristics.problem_instance import ProblemInstance


class ALNS:
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_inst = problem_instance

    def repair(self, schedule):
        solve_social_golfer(
                schedule,
                self.problem_inst.num_players,
                self.problem_inst.num_groups,
                self.problem_inst.num_weeks,
                max_time = 10
        )

    def destroy_weeks(self, schedule, destroy_weeks: int):
        """
        Remove x entire weeks from the schedule.

        Returns:
            new_schedule: schedule with destroyed weeks replaced by empty lists
        """

        max_num_weeks = len(schedule)
        destroy_weeks = min(destroy_weeks, max_num_weeks)

        # TODO maybe add that weeks with many conflicts are destroyed first?
        destroyed_week_ind = random.sample(range(max_num_weeks), destroy_weeks)

        new_schedule = []
        for w, week in enumerate(schedule):
            if w in destroyed_week_ind:
                # Preserve number of groups, but empty them
                new_schedule.append([[] for _ in range(self.problem_inst.num_groups)])
            else:
                new_schedule.append(week)

        return new_schedule


    def destroy_groups(self, schedule ,fraction: float):
        """
        Remove a fraction of all groups across all weeks.
        """
        all_groups = [
            (w, g)
            for w, week in enumerate(schedule)
            for g in range(len(week))
        ]

        k = int(len(all_groups) * fraction)
        destroyed = set(random.sample(all_groups, k))

        new_schedule = []
        for w, week in enumerate(schedule):
            new_week = []
            for g, group in enumerate(week):
                if (w, g) in destroyed:
                    new_week.append([])
                else:
                    new_week.append(group)
            new_schedule.append(new_week)

        return new_schedule

    @staticmethod
    def top_conflicting_pairs(conflicts, conf_threshold):
        """
        Return the x player pairs with the highest conflict counts (>1).
        """
        n = conflicts.shape[0]
        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                if conflicts[i, j] > 1:
                    pairs.append((conflicts[i, j], i, j))

        # Sort descending by conflict count
        pairs.sort(reverse=True, key=lambda t: t[0])

        return [(i, j) for _, i, j in pairs[:conf_threshold]]

    def destroy_conflicting_pairs(self, schedule, conflicts, conf_threshold):
        """
        Destroy all groups that contain any of the top most conflicting player pairs.

        Returns:
            new_schedule
        """
        pairs = ALNS.top_conflicting_pairs(conflicts, conf_threshold)
        pair_set = {frozenset(p) for p in pairs}

        new_schedule = []
        destroyed_locations = []

        for w, week in enumerate(schedule):
            new_week = []
            for g, group in enumerate(week):
                group_set = set(group)

                # Check if this group contains a bad pair
                destroy = False
                for i, j in pairs:
                    if i in group_set and j in group_set:
                        destroy = True
                        break

                if destroy:
                    new_week.append([])
                    destroyed_locations.append((w, g))
                else:
                    new_week.append(group)

            new_schedule.append(new_week)

        return new_schedule