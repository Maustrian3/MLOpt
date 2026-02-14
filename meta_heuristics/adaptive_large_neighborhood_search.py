import random
from typing import List, Optional

from meta_heuristics.problem_instance import ProblemInstance

try:
    from constraint_programming.exact_method import solve_social_golfer
except Exception:
    solve_social_golfer = None


class ALNS:
    """Adaptive Large Neighborhood Search helper.

    Provides destroy operators and an exact repair using CP-SAT.
    """
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_inst = problem_instance

    def repair(self, schedule: List[List[List[int]]]) -> Optional[List[List[List[int]]]]:
        """Repair a (possibly partial) schedule via exact CP-SAT."""
        if solve_social_golfer is None:
            return None

        return solve_social_golfer(
            partial_schedule=schedule,
            num_golfers=self.problem_inst.num_players,
            num_groups=self.problem_inst.num_groups,
            num_weeks=self.problem_inst.num_weeks,
            max_time=30,
            presolve=True,
            logging=False,
        )

    def destroy_weeks(self, schedule: List[List[List[int]]], destroy_weeks: int) -> List[List[List[int]]]:
        """Destroy operator: empty entire weeks."""
        max_num_weeks = len(schedule)
        destroy_weeks = min(destroy_weeks, max_num_weeks)
        destroyed_week_ind = set(random.sample(range(max_num_weeks), destroy_weeks))

        new_schedule: List[List[List[int]]] = []
        for w, week in enumerate(schedule):
            if w in destroyed_week_ind:
                new_schedule.append([[] for _ in range(self.problem_inst.num_groups)])
            else:
                new_schedule.append([group.copy() for group in week])
        return new_schedule

    def destroy_groups(self, schedule: List[List[List[int]]], fraction: float) -> List[List[List[int]]]:
        """Destroy operator: empty a fraction of all groups across all weeks."""
        all_groups = [(w, g) for w, week in enumerate(schedule) for g in range(len(week))]
        k = int(len(all_groups) * fraction)
        destroyed = set(random.sample(all_groups, k)) if k > 0 else set()

        new_schedule: List[List[List[int]]] = []
        for w, week in enumerate(schedule):
            new_week: List[List[int]] = []
            for g, group in enumerate(week):
                if (w, g) in destroyed:
                    new_week.append([])
                else:
                    new_week.append(group.copy())
            new_schedule.append(new_week)
        return new_schedule

    @staticmethod
    def top_conflicting_pairs(conflicts, conf_threshold: int):
        """Return up to conf_threshold player pairs with conflicts > 1."""
        n = conflicts.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if conflicts[i, j] > 1:
                    pairs.append((conflicts[i, j], i, j))
        pairs.sort(reverse=True, key=lambda t: t[0])
        return [(i, j) for _, i, j in pairs[:conf_threshold]]

    def destroy_conflicting_pairs(self, schedule, conflicts, conf_threshold: int):
        """Destroy operator: empty groups containing high conflict pairs."""
        if conflicts is None:
            return schedule

        pairs = self.top_conflicting_pairs(conflicts, conf_threshold)
        if not pairs:
            return schedule

        new_schedule = []
        for week in schedule:
            new_week = []
            for group in week:
                gset = set(group)
                destroy = any((i in gset and j in gset) for (i, j) in pairs)
                new_week.append([] if destroy else group.copy())
            new_schedule.append(new_week)
        return new_schedule
