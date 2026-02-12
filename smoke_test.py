"""
Smoke test to verify Task 1.1 is working correctly.
"""

from meta_heuristics.problem_instance import ProblemInstance
from meta_heuristics.solution_instance import SolutionInstance
from meta_heuristics.adaptive_large_neighborhood_search import ALNS
from meta_heuristics.MAB.alns_arm import ALNSArm
from meta_heuristics.MAB.multi_armed_bandit import ALNSMultiArmedBandit


def test_basic_repair():
    """Test that the repair operator works"""
    print("\n=== Test 1: Basic Repair ===")
    
    # Use FEASIBLE instance: 9 players, 3 groups, 4 weeks
    problem = ProblemInstance(num_groups=3, num_players=9, num_weeks=4)
    alns = ALNS(problem)
    
    # Create an empty schedule
    empty_schedule = [[[] for _ in range(3)] for _ in range(4)]
    
    print(f"Problem: {problem.num_players} players, {problem.num_groups} groups, {problem.num_weeks} weeks")
    print("Calling repair on empty schedule...")
    
    # Try to repair it
    repaired = alns.repair(empty_schedule)
    
    if repaired is None:
        print(" FAILED: Repair returned None")
        return False
    
    print(f"✅ PASSED: Repair returned a schedule")
    return True


def test_destroy_operators():
    """Test that destroy operators work"""
    print("\n=== Test 2: Destroy Operators ===")
    
    problem = ProblemInstance(num_groups=3, num_players=9, num_weeks=4)
    solution = SolutionInstance(problem, SolutionInstance.generate_random_solution)
    alns = ALNS(problem)
    
    # Test destroy weeks
    destroyed = alns.destroy_weeks(solution.schedule, 1)
    empty_weeks = sum(1 for week in destroyed if all(len(group) == 0 for group in week))
    
    if empty_weeks != 1:
        print(f" FAILED: destroy_weeks should empty 1 week, emptied {empty_weeks}")
        return False
    
    print(f"✅ PASSED: destroy_weeks works correctly")
    
    # Test destroy groups
    destroyed = alns.destroy_groups(solution.schedule, 0.3)
    print(f"✅ PASSED: destroy_groups works correctly")
    
    return True


def test_full_loop():
    """Test the full ALNS-MAB loop"""
    print("\n=== Test 3: Full ALNS-MAB Loop ===")
    
    problem = ProblemInstance(num_groups=3, num_players=9, num_weeks=4)
    solution = SolutionInstance(problem, SolutionInstance.generate_random_solution)
    
    print(f"Initial violations: {solution.violation_count}")
    
    alns_helper = ALNS(problem)
    
    # Create arms - operators work on SCHEDULES, not SolutionInstance objects
    # ADDED THE 3RD OPERATOR HERE
    arms = [
        ALNSArm("Destroy 1 Week", lambda sched: alns_helper.destroy_weeks(sched, 1)),
        ALNSArm("Destroy 30% Groups", lambda sched: alns_helper.destroy_groups(sched, 0.3)),
        ALNSArm("Destroy Conflicts", lambda sched: alns_helper.destroy_conflicting_pairs(sched, solution.conflicts, 1))
    ]
    
    # Run short test
    mab = ALNSMultiArmedBandit(
        arms=arms,
        solution_instance=solution,
        max_iterations=20,
        initial_temperature=50.0,
        cooling_rate=0.99,
        seed=42,
    )
    
    print(f"Running MAB loop with {len(arms)} arms...")
    best = mab.run(verbose=True)
    
    if best is None:
        print("FAILED: MAB returned None")
        return False
    
    print(f"PASSED: Full loop completed")
    print(f"   Best violations: {best.violation_count}")
    return True


if __name__ == "__main__":
    print("="*60)
    print("SMOKE TEST FOR TASK 1.1")
    print("="*60)
    
    test1 = test_basic_repair()
    test2 = test_destroy_operators()
    test3 = test_full_loop()
    
    print("\n" + "="*60)
    if test1 and test2 and test3:
        print(" ALL TESTS PASSED ")
        print("Task 1.1 plumbing is working with 3 operators!")
    else:
        print(" SOME TESTS FAILED")
        print("Check the error messages above")