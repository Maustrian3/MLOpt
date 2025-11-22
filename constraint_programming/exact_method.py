"""
Social Golfer Problem Solver using OR-Tools CP-SAT

Problem: Schedule golfers into groups over multiple weeks such that:
- Each group has the same number of players
- No two golfers play in the same group more than once
- All golfers play each week

Classic instance: 32 golfers, 8 groups of 4, over 10 weeks
"""

from ortools.sat.python import cp_model


def solve_social_golfer(num_golfers, num_groups, num_weeks):
    """
    Solve the social golfer problem.

    Args:
        num_golfers: Number of golfers
        num_groups: Number of groups per week
        num_weeks: Number of weeks to schedule

    Returns:
        Solution as a list of weeks, each containing groups of golfer IDs
    """
    group_size = int(num_golfers / num_groups)

    model = cp_model.CpModel()

    # Variables: play[w][g][p] = 1 if golfer p plays in group g in week w
    play = {}
    for w in range(num_weeks):
        for g in range(num_groups):
            for p in range(num_golfers):
                play[(w, g, p)] = model.NewBoolVar(f'play_w{w}_g{g}_p{p}')

    # Constraint 1: Each golfer plays exactly once per week
    for w in range(num_weeks):
        for p in range(num_golfers):
            model.Add(sum(play[(w, g, p)] for g in range(num_groups)) == 1) # The golfer only plays in one group

    # Constraint 2: Each group has exactly group_size golfers
    for w in range(num_weeks):
        for g in range(num_groups):
            model.Add(sum(play[(w, g, p)] for p in range(num_golfers)) == group_size) # The group has group_size golfers

    # Constraint 3: No two golfers meet more than once
    # For each golfer g1 and golfer g2
    for g1 in range(num_golfers):
        for g2 in range(g1 + 1, num_golfers):
            meet_vars = []
            for w in range(num_weeks):
                for g in range(num_groups):
                    meet = model.NewBoolVar(f'meet_p{g1}_p{g2}_w{w}_g{g}')
                    # meet = 1 iff both play[(w,g,g1)] and play[(w,g,g2)] are 1
                    # if play[(w, g, g1)] AND play[(w, g, g2)] -> meet = 1
                    model.AddBoolAnd([play[(w, g, g1)], play[(w, g, g2)]]).OnlyEnforceIf(meet)
                    # if NOT play[(w, g, g1) OR NOT play[(w, g, g1) -> meet = 0
                    model.AddBoolOr([play[(w, g, g1)].Not(), play[(w, g, g2)].Not()]).OnlyEnforceIf(meet.Not())
                    meet_vars.append(meet)

            # Sum of meetings must be at most 1
            # So they can meet 0 or 1 times
            model.Add(sum(meet_vars) <= 1)

    # TODO Breaking symmetry reduces possible solution which come from combinatorial permutation which lead to the same solution
    #  Maybe measure impact of symmetry breaking, would be interesting?
    # Symmetry breaking: Fix first week to reduce search space
    # Assign golfers 0..group_size-1 to group 0, etc.
    for g in range(num_groups):
        for p in range(g * group_size, (g + 1) * group_size):
            model.Add(play[(0, g, p)] == 1)

    # Symmetry breaking: Golfer 0 is always in group 0
    for w in range(num_weeks):
        model.Add(play[(w, 0, 0)] == 1)

    # Create solver and solve
    solver = cp_model.CpSolver()
    # solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.log_search_progress = True

    print(f"Solving Social Golfer Problem:")
    print(f"  {num_golfers} golfers, {num_groups} groups of {group_size}, {num_weeks} weeks\n")

    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"\nSolution found!")
        print(f"Status: {solver.StatusName(status)}")
        print(f"Time: {solver.WallTime():.2f}s\n")

        # Extract solution
        schedule = []
        for w in range(num_weeks):
            week = []
            for g in range(num_groups):
                group = []
                for p in range(num_golfers):
                    if solver.Value(play[(w, g, p)]) == 1:
                        group.append(p)
                week.append(sorted(group))
            schedule.append(week)

        print("Schedule:")
        for w, week in enumerate(schedule):
            print(f"\nWeek {w + 1}:")
            for g, group in enumerate(week):
                print(f"  Group {g + 1}: {group}")

        verify_solution(schedule, num_golfers, group_size)

        return schedule
    else:
        print(f"\nNo solution found.")
        print(f"Status: {solver.StatusName(status)}")
        return None


# TODO Use verification also in meta heuristic
def verify_solution(schedule, num_golfers, group_size):
    """Verify that the solution satisfies all constraints."""
    print("Verification:")

    num_weeks = len(schedule)
    num_groups = len(schedule[0])

    # Check group sizes
    for w, week in enumerate(schedule):
        for g, group in enumerate(week):
            assert len(group) == group_size, f"Week {w + 1}, Group {g + 1} has wrong size"

    print(f"All {len(schedule)} weeks valid")

    # Check each golfer plays once per week
    for w, week in enumerate(schedule):
        golfers_this_week = set()
        for group in week:
            golfers_this_week.update(group)
        assert len(golfers_this_week) == num_golfers, f"Week {w + 1} missing golfers"

    print(f"All golfers play exactly once per week")

    # Check no two golfers meet twice
    meetings = {}
    for w, week in enumerate(schedule):
        for group in week:
            for i, p1 in enumerate(group):
                for p2 in group[i + 1:]:
                    pair = tuple(sorted([p1, p2]))
                    if pair in meetings:
                        print(f"Golfers {p1} and {p2} met in weeks {meetings[pair] + 1} and {w + 1}")
                        return False
                    meetings[pair] = w

    print(f"No pair of golfers meets more than once")

    return True


if __name__ == "__main__":
    solve_social_golfer(num_golfers=32, num_groups=8, num_weeks=6)