def verify_solution(schedule, num_golfers, group_size, logging: bool = False) -> bool:
    """Verify that the solution satisfies all constraints.

    Returns:
        True if the schedule satisfies constraints, False otherwise.
    """
    if logging:
        print("Verification:")

    num_weeks = len(schedule)
    num_groups = len(schedule[0]) if num_weeks > 0 else 0

    # Check group sizes
    for w, week in enumerate(schedule):
        for g, group in enumerate(week):
            if len(group) != group_size:
                if logging:
                    print(f"Week {w + 1}, Group {g + 1} has wrong size: {len(group)} != {group_size}")
                return False

    # Check each golfer plays once per week
    for w, week in enumerate(schedule):
        golfers_this_week = set()
        for group in week:
            golfers_this_week.update(group)
        if len(golfers_this_week) != num_golfers:
            if logging:
                print(f"Week {w + 1} missing golfers: {len(golfers_this_week)} != {num_golfers}")
            return False

    # Check no two golfers meet twice
    meetings = {}
    for w, week in enumerate(schedule):
        for group in week:
            for i, p1 in enumerate(group):
                for p2 in group[i + 1:]:
                    pair = tuple(sorted((p1, p2)))
                    if pair in meetings:
                        if logging:
                            print(f"Golfers {p1} and {p2} met in weeks {meetings[pair] + 1} and {w + 1}")
                        return False
                    meetings[pair] = w

    return True
