"""
Compare Task 1.1 (MAB) vs Task 1.2 (RL)
"""

import csv
import pandas as pd


def compare():
    # Load results
    mab = pd.read_csv('task1_1_results.csv')
    rl = pd.read_csv('task1_2_results.csv')
    
    print("="*60)
    print("TASK 1.1 (MAB) vs TASK 1.2 (RL)")
    print("="*60)
    
    print(f"\nMAB Results:")
    print(f"  Instances: {len(mab)}")
    print(f"  Avg improvement: {mab['improvement'].mean():.2f}")
    print(f"  Avg final violations: {mab['final_violations'].mean():.2f}")
    print(f"  Avg time: {mab['time_seconds'].mean():.2f}s")
    
    print(f"\nRL Results:")
    print(f"  Instances: {len(rl)}")
    print(f"  Avg improvement: {rl['improvement'].mean():.2f}")
    print(f"  Avg final violations: {rl['final_violations'].mean():.2f}")
    print(f"  Avg time: {rl['time_seconds'].mean():.2f}s")
    
    # Winner
    mab_better = (mab['final_violations'] < rl['final_violations']).sum() if len(mab) == len(rl) else 0
    print(f"\n📊 Head-to-head: MAB won {mab_better}/{len(rl)} instances")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    compare()