# TODO use verification also for meta heuristic solution
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