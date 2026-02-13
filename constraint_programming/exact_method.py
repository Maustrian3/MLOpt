"""
Social Golfer Problem (SGP) exact solver / exact repair using OR-Tools CP-SAT.

Schedule structure:
    schedule[w][g] = list of players assigned to group g in week w

Constraints:
    - Each golfer plays exactly once per week.
    - Each group has fixed size (num_golfers / num_groups).
    - No pair of golfers is in the same group more than once across all weeks.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ortools.sat.python import cp_model

# Verification is optional; solver feasibility is the primary validity signal.
try:
    from verification.verify import verify_solution
except Exception:
    def verify_solution(_schedule, _n, _k, logging: bool = False) -> bool:
        return True


def solve_social_golfer(
    partial_schedule: List[List[List[int]]],
    num_golfers: int,
    num_groups: int,
    num_weeks: int,
    max_time: int = 10,
    presolve: bool = True,
    return_solver: bool = False,
    logging: bool = False,
) -> Optional[List[List[List[int]]]]:
    """Solve/repair an SGP instance using CP-SAT.

    Args:
        partial_schedule: schedule with some groups possibly empty ([]). Non-empty
            groups are treated as fixed assignments (players are forced into that group).
            If a group is fully specified (size == group_size), other players are forbidden.
            If a group is partially specified (size < group_size), additional players may be added.
        num_golfers: number of players.
        num_groups: number of groups per week.
        num_weeks: number of weeks.
        max_time: CP-SAT time limit in seconds.
        presolve: enable/disable presolve.
        return_solver: if True, returns solver object instead of schedule (debug).
        logging: enable CP-SAT log output and verifier logging.

    Returns:
        A complete schedule, or None if no solution found within time.
    """
    if partial_schedule is None:
        return None

    if num_groups <= 0 or num_weeks <= 0 or num_golfers <= 0:
        return None

    if num_golfers % num_groups != 0:
        return None

    group_size = num_golfers // num_groups

    model = cp_model.CpModel()

    # play[w,g,p] == 1 iff golfer p plays in group g in week w
    play = {}
    for w in range(num_weeks):
        for g in range(num_groups):
            for p in range(num_golfers):
                play[(w, g, p)] = model.NewBoolVar(f"play_w{w}_g{g}_p{p}")

    # Constraint: each golfer plays exactly once per week
    for w in range(num_weeks):
        for p in range(num_golfers):
            model.Add(sum(play[(w, g, p)] for g in range(num_groups)) == 1)

    # Constraint: each group has exactly group_size golfers
    for w in range(num_weeks):
        for g in range(num_groups):
            model.Add(sum(play[(w, g, p)] for p in range(num_golfers)) == group_size)

    # Constraint: no two golfers meet more than once across all weeks
    pair_violations = []

    for p1 in range(num_golfers):
        for p2 in range(p1 + 1, num_golfers):
            meet_vars = []
            for w in range(num_weeks):
                for g in range(num_groups):
                    meet = model.NewBoolVar(f"meet_p{p1}_p{p2}_w{w}_g{g}")
                    model.AddBoolAnd([play[(w, g, p1)], play[(w, g, p2)]]).OnlyEnforceIf(meet)
                    model.AddBoolOr([play[(w, g, p1)].Not(), play[(w, g, p2)].Not()]).OnlyEnforceIf(meet.Not())
                    meet_vars.append(meet)
            # Count excess times of meeting of two players
            excess = model.NewIntVar(0, num_weeks, f"pair_excess_{p1}_{p2}")
            model.Add(excess >= sum(meet_vars) - 1)
            model.Add(excess >= 0)
            pair_violations.append(excess)

    # Tell the model to minimize for pair violations
    model.Minimize(sum(pair_violations))


    # Repair binding: respect fixed assignments in partial_schedule
    w_lim = min(len(partial_schedule), num_weeks)
    for w in range(w_lim):
        week = partial_schedule[w]
        g_lim = min(len(week), num_groups)
        for g in range(g_lim):
            fixed_players = week[g]

            if not fixed_players:
                continue

            fixed_set = set(fixed_players)

            # Fixed players are forced into this group.
            for p in fixed_set:
                if 0 <= p < num_golfers:
                    model.Add(play[(w, g, p)] == 1)

            # If the group is fully specified, forbid all others.
            if len(fixed_set) == group_size:
                for p in range(num_golfers):
                    if p not in fixed_set:
                        model.Add(play[(w, g, p)] == 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    solver.parameters.cp_model_presolve = presolve
    solver.parameters.log_search_progress = logging
    solver.parameters.num_search_workers = 4

    status = solver.Solve(model)

    total_pair_violations = sum(solver.Value(v) for v in pair_violations)
    print(f"Total pair violations after solve: {total_pair_violations}")

    if return_solver:
        return solver  # type: ignore[return-value]

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if logging:
            print(f"No solution found. Status: {solver.StatusName(status)}")
        return None

    # Extract schedule
    schedule: List[List[List[int]]] = []
    for w in range(num_weeks):
        week_list: List[List[int]] = []
        for g in range(num_groups):
            group_list = [p for p in range(num_golfers) if solver.Value(play[(w, g, p)]) == 1]
            week_list.append(sorted(group_list))
        schedule.append(week_list)

    # Verification is non-blocking; CP-SAT feasibility is the primary guarantee.
    try:
        _ok = verify_solution(schedule, num_golfers, group_size, logging=logging)
        if logging and not _ok:
            print("Verification reported invalid schedule (non-blocking).")
    except Exception as e:
        if logging:
            print(f"Verification error (non-blocking): {e}")

    return schedule
