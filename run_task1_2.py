"""
Task 1.2: RL-ALNS
Train policy on 70% instances, test on 30%
FIXED: Policy persists across all training instances
"""

import csv
import time
import random
import torch
from meta_heuristics.problem_instance import ProblemInstance
from meta_heuristics.RL.dr_alns_env import DRALNSEnv
from meta_heuristics.RL.dr_alns_policy import DRALNSPolicy
from meta_heuristics.RL.dr_alns import collect_trajectory, compute_returns, ppo_update


def train_policy_on_multiple_instances(train_instances, epochs_per_instance=200):
    """
    Train ONE policy across multiple instances.
    """
    print(f"Training RL policy on {len(train_instances)} instances...")
    
    # Create ONE policy (shared across ALL instances)
    policy = DRALNSPolicy(state_dim=7, num_actions=3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    # Train for multiple epochs over all instances
    for epoch in range(3):  # 3 passes over entire training set
        print(f"\n=== Training Epoch {epoch+1}/3 ===")
        
        for i, (g, p, w) in enumerate(train_instances, 1):
            print(f"  [{i}/{len(train_instances)}] Training on {g}-{p}-{w}...")
            
            problem = ProblemInstance(g, p, w)
            env = DRALNSEnv(problem, max_steps=100)
            
            # Train the SHARED policy on this environment
            for _ in range(epochs_per_instance):
                # Collect trajectory
                states, actions, rewards, log_probs, values = collect_trajectory(
                    env, policy, horizon=50
                )
                
                # Compute returns
                returns = compute_returns(rewards)
                
                # Update policy (weights accumulate!)
                ppo_update(
                    policy, optimizer,
                    states, actions, log_probs,
                    returns, values
                )
    
    # Save the trained policy
    torch.save(policy.state_dict(), 'rl_policy.pth')
    print("\n✅ Policy trained and saved to rl_policy.pth")
    
    return policy


def test_policy_on_instance(policy, groups, players, weeks, max_steps=200):
    """Test trained policy on one instance"""
    print(f"[RL] Testing {groups}-{players}-{weeks}...", end=" ")
    
    try:
        problem = ProblemInstance(groups, players, weeks)
        env = DRALNSEnv(problem, max_steps)
        
        state = env.reset()
        initial = env.solution.violation_count
        
        start = time.time()
        
        # Run policy (greedy  no exploration)
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
    COMPLETE main function with all steps:
    1. Load instances
    2. Split 70/30
    3. Train on 70%
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
    print(f"TASK 1.2: RL-ALNS")
    print(f"  Train set: {len(train_instances)} instances")
    print(f"  Test set: {len(test_instances)} instances")
    print(f"{'='*60}")
    
    # STEP 1: Train (policy persists across all instances)
    policy = train_policy_on_multiple_instances(train_instances, epochs_per_instance=200)
    
    # STEP 2: Test on unseen instances
    print(f"\n{'='*60}")
    print(f"Testing trained policy on {len(test_instances)} instances")
    print(f"{'='*60}")
    
    results = []
    for i, (g, p, w) in enumerate(test_instances, 1):
        print(f"[{i}/{len(test_instances)}]", end=" ")
        result = test_policy_on_instance(policy, g, p, w, max_steps=100)
        if result:
            results.append(result)
    
    # STEP 3: Save results
    if results:
        with open('task1_2_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'='*60}")
        print(f"✅ Task 1.2 Complete! Tested {len(results)}/{len(test_instances)}")
        print(f"   Results -> task1_2_results.csv")
        print(f"{'='*60}")
    else:
        print("\n❌ No results to save!")


if __name__ == "__main__":
    main()