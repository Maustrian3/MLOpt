# csv_helper.py
"""
Generate 100+ Social Golfer Problem instances for experiments.
Mix of easy, moderate, and hard instances including the original 8-4-x problem.
"""

import csv

SGP_INSTANCES = [
    # =================================================================
    # CRITICAL: Original Problem Variants (8-4-x) - MUST INCLUDE
    # =================================================================
    (8, 32, 5),   # 8-4-5: Moderately hard
    (8, 32, 6),   # 8-4-6: Hard
    (8, 32, 7),   # 8-4-7: Very hard
    (8, 32, 8),   # 8-4-8: Extremely hard
    (8, 32, 9),   # 8-4-9: Near impossible
    (8, 32, 10),  # 8-4-10: Maximum known (may timeout)
    
    # =================================================================
    # TIER 1: EASY (Small instances, quick to solve)
    # =================================================================
    (2, 6, 2),    (2, 6, 3),    (2, 6, 4),
    (2, 8, 2),    (2, 8, 3),    
    (3, 9, 2),    (3, 9, 3),    (3, 9, 4),
    (3, 12, 2),   (3, 12, 3),   (3, 12, 4),
    (4, 12, 2),   (4, 12, 3),   (4, 12, 4),
    (4, 16, 2),   (4, 16, 3),   (4, 16, 4),
    
    # =================================================================
    # TIER 2: MODERATE 
    # =================================================================
    (4, 12, 5),   (4, 16, 5),   
    (5, 15, 2),   (5, 15, 3),   (5, 15, 4),   (5, 15, 5),
    (5, 20, 2),   (5, 20, 3),   (5, 20, 4),   
    (6, 18, 2),   (6, 18, 3),   (6, 18, 4),   
    (6, 24, 2),   (6, 24, 3),   
    (7, 21, 2),   (7, 21, 3),   (7, 21, 4),
    (7, 28, 2),   (7, 28, 3),
    
    # =================================================================
    # TIER 3: HARD (Larger instances)
    # =================================================================
    (5, 15, 6),   (5, 20, 5),   (6, 18, 5),   
    (6, 24, 4),   (6, 24, 5),
    (7, 21, 5),   (7, 28, 4),
    (8, 24, 3),   (8, 24, 4),   (8, 24, 5),
    
    # =================================================================
    # Additional variations for 100+ total
    # =================================================================
    (3, 15, 2),   (3, 15, 3),   (3, 15, 4),
    (4, 20, 2),   (4, 20, 3),   (4, 20, 4),
    (5, 25, 2),   (5, 25, 3),   (5, 25, 4),
    (6, 30, 2),   (6, 30, 3),
    (7, 35, 2),   (7, 35, 3),
    (9, 27, 2),   (9, 27, 3),   (9, 27, 4),
    (9, 36, 2),   (9, 36, 3),
    (10, 30, 2),  (10, 30, 3),
    (10, 40, 2),  (10, 40, 3),
    
    # More group size 4 variants (common in literature)
    (4, 8, 2),    (4, 8, 3),    (4, 8, 4),    (4, 8, 5),
    (5, 10, 2),   (5, 10, 3),   (5, 10, 4),
    (6, 12, 2),   (6, 12, 3),   (6, 12, 4),
    (7, 14, 2),   (7, 14, 3),
    (8, 16, 2),   (8, 16, 3),   (8, 16, 4),
    (9, 18, 2),   (9, 18, 3),
    (10, 20, 2),  (10, 20, 3),
    
    # Even more to reach 100+
    (2, 10, 2),   (2, 10, 3),
    (3, 6, 2),    (3, 6, 3),    (3, 6, 4),
    (11, 33, 2),  (11, 33, 3),
    (12, 36, 2),  (12, 36, 3),
]

def generate_instances_csv(filename="sgp_all_instances.csv"):
    """Generate CSV with all instances"""
    
    # Remove duplicates and sort
    unique_instances = sorted(set(SGP_INSTANCES))
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["num_groups", "num_players", "num_weeks"])
        writer.writerows(unique_instances)
    
    print(f"✅ Generated {len(unique_instances)} instances")
    print(f"   Saved to: {filename}")
    
    # Statistics
    tier1 = [inst for inst in unique_instances if inst[2] <= 3 and inst[1] <= 16]
    tier2 = [inst for inst in unique_instances if 4 <= inst[2] <= 5 and inst[1] <= 24]
    tier3 = [inst for inst in unique_instances if inst[2] >= 6 or inst[1] >= 28]
    original = [inst for inst in unique_instances if inst[0] == 8 and inst[1] == 32]
    
    print(f"\n   Easy instances: {len(tier1)}")
    print(f"   Moderate instances: {len(tier2)}")
    print(f"   Hard instances: {len(tier3)}")
    print(f"   Original 8-4-x instances: {len(original)}")
    
    return unique_instances

if __name__ == "__main__":
    generate_instances_csv()