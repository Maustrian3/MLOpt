"""
Social Golfer Problem Solver using OR-Tools CP-SAT

Problem: Schedule golfers into groups over multiple weeks such that:
- Each group has the same number of players
- No two golfers play in the same group more than once
- All golfers play each week

Classic instance: 32 golfers, 8 groups of 4, over 10 weeks
"""

from ortools.sat.python import cp_model

from verficiation.verify import verify_solution


def solve_social_golfer(num_golfers, num_groups, num_weeks, max_time,
                        presolve = True,return_solver: bool = False, logging: bool = False):

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
    solver.parameters.max_time_in_seconds = max_time
    solver.parameters.log_search_progress = logging

    if logging:
        print(f"Solving Social Golfer Problem:")
        print(f"  {num_golfers} golfers, {num_groups} groups of {group_size}, {num_weeks} weeks\n")

    solver.parameters.cp_model_presolve = presolve

    status = solver.Solve(model)

    if return_solver:
        return solver

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if logging:
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

        if logging:
            print("Schedule:")
            for w, week in enumerate(schedule):
                print(f"\nWeek {w + 1}:")
                for g, group in enumerate(week):
                    print(f"  Group {g + 1}: {group}")

        if not verify_solution(schedule, num_golfers, group_size):
            if logging:
                print(f"\nSolution invalid.")
                return None

        return schedule
    else:
        if logging:
            print(f"\nNo solution found.")
            print(f"Status: {solver.StatusName(status)}")
            return None


if __name__ == "__main__":
    solver = solve_social_golfer(num_golfers=32,
                                 num_groups=8,
                                 num_weeks=6,
                                 max_time=3,
                                 presolve=False,
                                 return_solver=True,
                                 logging=True
                                 )
    print("-"*30)
    print(solver.ResponseStats())
    print("+"*30)
    print(solver.ResponseProto().num_integer_propagations)
