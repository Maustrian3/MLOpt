import csv
import random

# Output filename
outfile = "SGP_test_instances.csv"

# Number of random instances for each week value
num_instances_per_week = 5

# Input search ranges
num_golfers = range(8, 32)
num_weeks = range(2, 9)
num_groups = range(2, 13)

# Storage for final picked instances
final_instances = []

for week in num_weeks:

    # Collect all valid instances for this specific W
    valid_for_week = []

    for golfer in num_golfers:
        for group in num_groups:
            golfers_per_group = golfer // group

            # groups must divide golfers
            if golfer % group != 0:
                continue

            # Each golfer plays with (golfers_per_group − 1) other golfers per week.
            # Across W weeks, they meet at most: [weeks * (golfers_per_group - 1)]
            # So there must be more group capacity than golfers [weeks * (golfers_per_group – 1)] ≥ [(golfers – 1)]
            if week * (golfers_per_group - 1) >= (golfer - 1):
                continue

            # skip nonsense
            if week > golfer:
                continue

            valid_for_week.append((golfer, week, group))

    # Randomly sample exactly Y instances OR all if fewer available
    if len(valid_for_week) <= num_instances_per_week:
        chosen = valid_for_week
    else:
        chosen = random.sample(valid_for_week, num_instances_per_week)

    final_instances.extend(chosen)

# Write CSV file
with open(outfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["num_golfers", "num_weeks", "num_groups"])
    writer.writerows(final_instances)

print(f"Generated {len(final_instances)} instances (Y={num_instances_per_week} per week) and saved to {outfile}")
