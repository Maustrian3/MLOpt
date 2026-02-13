"""
Task 1.1: Run MAB-ALNS on all instances
Each instance gets a FRESH Multi-Armed Bandit
"""

import csv
import time
from meta_heuristics.problem_instance import ProblemInstance
from meta_heuristics.solution_instance import SolutionInstance
from meta_heuristics.adaptive_large_neighborhood_search import ALNS
from meta_heuristics.MAB.alns_arm import ALNSArm
from meta_heuristics.MAB.multi_armed_bandit import ALNSMultiArmedBandit


def run_mab_instance(groups, players, weeks, max_iter=100, seed=42):
    """Run MAB-ALNS on one instance"""
    print(f"\n[MAB] {groups}-{players}-{weeks}...", end=" ")
    
    try:
        problem = ProblemInstance(groups, players, weeks)
        solution = SolutionInstance(problem, SolutionInstance.generate_random_solution)
        initial = solution.violation_count
        
        alns = ALNS(problem)
        
        # 3 arms using SolutionInstance methods
        arms = [
            ALNSArm("Destroy 2 Weeks", lambda s: alns.destroy_weeks(s, 2)),
            ALNSArm("Destroy 30% Groups", lambda s: alns.destroy_groups(s, 0.3)),
            ALNSArm("Destroy Conflicts", lambda s: alns.destroy_conflicting_pairs(s, solution.conflicts, 2)),
        ]
        
        # Fresh MAB for this instance
        mab = ALNSMultiArmedBandit(
            arms=arms,
            solution_instance=solution,
            max_iterations=max_iter,
            initial_temperature=100.0,
            cooling_rate=0.99,
            seed=seed
        )
        
        start = time.time()
        best = mab.run(verbose=False)
        elapsed = time.time() - start
        
        final = best.violation_count if best else initial
        improvement = initial - final
        
        print(f"{initial}→{final} (Δ={improvement}) in {elapsed:.1f}s")
        
        return {
            'num_groups': groups,
            'num_players': players,
            'num_weeks': weeks,
            'method': 'MAB',
            'initial_violations': initial,
            'final_violations': final,
            'improvement': improvement,
            'time_seconds': round(elapsed, 2),
        }
    
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    instances = []
    with open('sgp_all_instances.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instances.append((int(row['num_groups']), int(row['num_players']), int(row['num_weeks'])))
    
    print(f"{'='*60}")
    print(f"TASK 1.1: MAB-ALNS on {len(instances)} instances")
    print(f"{'='*60}")
    
    results = []
    for i, (g, p, w) in enumerate(instances, 1):
        print(f"[{i}/{len(instances)}]", end=" ")
        result = run_mab_instance(g, p, w, max_iter=100)
        if result:
            results.append(result)
    
    # Save
    with open('task1_1_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*60}")
    print(f"✅ Task 1.1 Complete! Solved {len(results)}/{len(instances)}")
    print(f"   Results -> task1_1_results.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()