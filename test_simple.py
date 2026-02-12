"""Simple test with a FEASIBLE instance"""

from meta_heuristics.problem_instance import ProblemInstance
from constraint_programming.exact_method import solve_social_golfer

print("="*60)
print("SIMPLE REPAIR TEST - FEASIBLE INSTANCE")
print("="*60)

# FEASIBLE instance: 9 players, 3 groups of 3, 4 weeks
num_groups = 3
num_players = 9
num_weeks = 4
group_size = num_players // num_groups

print(f"\nProblem: {num_players} players, {num_groups} groups of {group_size}, {num_weeks} weeks")

# Empty schedule
empty = [[[] for _ in range(num_groups)] for _ in range(num_weeks)]

print(f"\nCalling solve_social_golfer...")

import time
start = time.time()

result = solve_social_golfer(
    partial_schedule=empty,
    num_golfers=num_players,
    num_groups=num_groups,
    num_weeks=num_weeks,
    max_time=30,
    presolve=True,
    logging=False,  # Turn off verbose logging
)

elapsed = time.time() - start

print(f"Repair took {elapsed:.2f} seconds")

if result is None:
    print(" FAILED: Repair returned None")
else:
    print(" SUCCESS: Got solution!")
    for w, week in enumerate(result):
        print(f"  Week {w}: {week}")