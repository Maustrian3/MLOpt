import csv
import random

# Output filename
outfile = "SGP_test_instances.csv"

# Number of random instances for each week value
num_instances_per_week = 15

# Input search ranges
num_golfers = range(8, 33)  # Because the classic SGP includes 32 golfers, we extend the range to include 32 -Kaan
num_weeks = range(2, 11)    # Because the required original family uses weeks 6..10, we extend weeks to include 10 -Kaan
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

# Because the project explicitly requires the original SGP family (32 golfers, 8 groups of 4, weeks 6..10),
# we force-add these instances even if random sampling misses them -Kaan
required_original_family = [(32, w, 8) for w in range(6, 11)]
final_instances.extend(required_original_family)

# Because duplicates can appear (random sampling might already include some required rows),
# we de-duplicate to keep one row per instance triple -Kaan
final_instances = list(set(final_instances))

# Because the CSV is easier to inspect/debug when ordered, we sort by (weeks, golfers, groups) -Kaan
final_instances.sort(key=lambda t: (t[1], t[0], t[2]))

# Write CSV file
with open(outfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["num_golfers", "num_weeks", "num_groups"])
    writer.writerows(final_instances)

print(f"Generated {len(final_instances)} instances (Y={num_instances_per_week} per week) and saved to {outfile}")
