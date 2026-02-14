"""
Compare MAB vs RL and generate plots for slides
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison():
    # Load results
    mab = pd.read_csv('task1_1_results.csv')
    rl = pd.read_csv('task1_2_results.csv')
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Average improvement
    data = pd.DataFrame({
        'MAB': [mab['improvement'].mean()],
        'RL': [rl['improvement'].mean()]
    })
    data.T.plot(kind='bar', ax=axes[0], legend=False, color=['#2E86AB', '#A23B72'])
    axes[0].set_title('Average Improvement', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Violations Reduced')
    axes[0].set_xlabel('')
    
    # Plot 2: Average time
    data = pd.DataFrame({
        'MAB': [mab['time_seconds'].mean()],
        'RL': [rl['time_seconds'].mean()]
    })
    data.T.plot(kind='bar', ax=axes[1], legend=False, color=['#2E86AB', '#A23B72'])
    axes[1].set_title('Average Runtime', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Seconds')
    axes[1].set_xlabel('')
    
    # Plot 3: Final violations distribution
    axes[2].hist(mab['final_violations'], bins=20, alpha=0.6, label='MAB', color='#2E86AB')
    axes[2].hist(rl['final_violations'], bins=20, alpha=0.6, label='RL', color='#A23B72')
    axes[2].set_title('Final Violations Distribution', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Final Violations')
    axes[2].set_ylabel('Count')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
    print("✅ Plot saved to comparison_plot.png")
    
    # Print summary
    print("\n" + "="*60)
    print("MAB vs RL COMPARISON")
    print("="*60)
    print(f"\nMAB: {len(mab)} instances")
    print(f"  Avg improvement: {mab['improvement'].mean():.2f}")
    print(f"  Avg time: {mab['time_seconds'].mean():.2f}s")
    
    print(f"\nRL: {len(rl)} instances")
    print(f"  Avg improvement: {rl['improvement'].mean():.2f}")
    print(f"  Avg time: {rl['time_seconds'].mean():.2f}s")

if __name__ == "__main__":
    plot_comparison()