"""
Task 1.2: RL-ALNS
Train policy on 70% instances, test on 30%
OPTIMIZED: 2 outer loops, 3 inner epochs, horizon 20
"""

import csv
import time
import random
import torch
from meta_heuristics.problem_instance import ProblemInstance
from meta_heuristics.RL.dr_alns_env import DRALNSEnv
from meta_heuristics.RL.dr_alns_policy import DRALNSPolicy
from meta_heuristics.RL.dr_alns import collect_trajectory, compute_returns, ppo_update


def train_policy_on_multiple_instances(train_instances, epochs_per_instance=3):
    """
    Train ONE policy across multiple instances.
    
    OPTIMIZED SETTINGS:
    - 2 outer loops (see every problem twice)
    - 3 epochs per instance (breadth over depth)
    - horizon=20 (reduced from 50)
    
    """
    print(f"Training RL policy on {len(train_instances)} instances...")
    print(f"Config: 2 outer loops, {epochs_per_instance} epochs/instance, horizon=20")
    
    # Create ONE policy (shared across ALL instances)
    policy = DRALNSPolicy(state_dim=7, num_actions=3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    # 2 passes over entire training set (better generalization)
    for epoch in range(2):  # CHANGED from 3 to 2
        print(f"\n{'='*60}")
        print(f"OUTER LOOP {epoch+1}/2")
        print(f"{'='*60}")
        
        for i, (g, p, w) in enumerate(train_instances, 1):
            print(f"  [{i}/{len(train_instances)}] Training on {g}-{p}-{w}...", end=" ")
            
            problem = ProblemInstance(g, p, w)
            env = DRALNSEnv(problem, max_steps=100)
            
            # Train with 3 updates per instance (reduced from 200)
            for ep in range(epochs_per_instance):  #  CHANGED to parameter (3)
                # Collect trajectory with reduced horizon
                states, actions, rewards, log_probs, values = collect_trajectory(
                    env, policy, horizon=20  #  CHANGED from 50 to 20
                )
                
                # Compute returns
                returns = compute_returns(rewards)
                
                # Update policy (weights accumulate)
                ppo_update(
                    policy, optimizer,
                    states, actions, log_probs,
                    returns, values
                )
            
            # Progress indicator
            total_reward = sum(r.item() for r in rewards) if rewards else 0
            print(f"(reward: {total_reward:.1f})")
    
    # Save the trained policy
    torch.save(policy.state_dict(), 'rl_policy.pth')
    print(f"\n{'='*60}")
    print(f"✅ Policy trained and saved to rl_policy.pth")
    print(f"{'='*60}")
    
    return policy


def test_policy_on_instance(policy, groups, players, weeks, max_steps=100):
    """Test trained policy on one instance"""
    print(f"[RL] Testing {groups}-{players}-{weeks}...", end=" ")
    
    try:
        problem = ProblemInstance(groups, players, weeks)
        env = DRALNSEnv(problem, max_steps)
        
        state = env.reset()
        initial = env.solution.violation_count
        
        start = time.time()
        
        # Run policy (greedy no exploration)
        for step in range(max_steps):
            with torch.no_grad():
                logits, _ = policy(state)
                action = torch.argmax(logits).item()
            
            state, reward, done = env.step(action)
            
            if done:
                break
        
        elapsed = time.time() - start
        
        final = env.best_violations
        improvement = initial - final
        
        print(f"{initial}→{final} (Δ={improvement}) in {elapsed:.1f}s")
        
        return {
            'num_groups': groups,
            'num_players': players,
            'num_weeks': weeks,
            'method': 'RL',
            'initial_violations': initial,
            'final_violations': final,
            'improvement': improvement,
            'time_seconds': round(elapsed, 2),
        }
    
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    """
    Main execution:
    1. Load instances from CSV
    2. Split 70/30 train/test
    3. Train policy on 70% (2 passes)
    4. Test on 30%
    5. Save results
    """
    
    # Load instances
    instances = []
    with open('sgp_all_instances.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instances.append((
                int(row['num_groups']),
                int(row['num_players']),
                int(row['num_weeks'])
            ))
    
    # Split 70/30
    random.seed(42)
    random.shuffle(instances)
    split = int(len(instances) * 0.7)
    train_instances = instances[:split]
    test_instances = instances[split:]
    
    print(f"{'='*60}")
    print(f"TASK 1.2: RL-ALNS (OPTIMIZED)")
    print(f"{'='*60}")
    print(f"Total instances: {len(instances)}")
    print(f"Train set: {len(train_instances)} (70%)")
    print(f"Test set: {len(test_instances)} (30%)")
    print(f"\nEstimated training time: ~2.3 hours")
    print(f"{'='*60}")
    
    # STEP 1: Train policy
    start_training = time.time()
    policy = train_policy_on_multiple_instances(
        train_instances, 
        epochs_per_instance=3  #  OPTIMIZED
    )
    training_time = time.time() - start_training
    
    print(f"\n⏱️  Training completed in {training_time/60:.1f} minutes")
    
    # STEP 2: Test on unseen instances
    print(f"\n{'='*60}")
    print(f"TESTING on {len(test_instances)} unseen instances")
    print(f"{'='*60}")
    
    results = []
    for i, (g, p, w) in enumerate(test_instances, 1):
        print(f"[{i}/{len(test_instances)}] ", end="")
        result = test_policy_on_instance(policy, g, p, w, max_steps=100)
        if result:
            results.append(result)
    
    # STEP 3: Save results
    if results:
        with open('task1_2_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        # Summary statistics
        avg_improvement = sum(r['improvement'] for r in results) / len(results)
        perfect = sum(1 for r in results if r['final_violations'] == 0)
        
        print(f"\n{'='*60}")
        print(f"✅ TASK 1.2 COMPLETE!")
        print(f"{'='*60}")
        print(f"Tested: {len(results)}/{len(test_instances)} instances")
        print(f"Perfect solutions (0 violations): {perfect} ({perfect/len(results)*100:.1f}%)")
        print(f"Average improvement: {avg_improvement:.1f} violations")
        print(f"\nResults saved to: task1_2_results.csv")
        print(f"{'='*60}")
    else:
        print("\n❌ No results to save!")


if __name__ == "__main__":
    main()
