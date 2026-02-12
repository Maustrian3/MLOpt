import math
import random
from typing import Optional

from meta_heuristics.MAB.alns_arm import ALNSArm
from meta_heuristics.solution_instance import SolutionInstance


class ALNSMultiArmedBandit:
    """ALNS loop with bandit-based destroy-operator selection (roulette wheel)."""

    def __init__(
        self,
        arms: list[ALNSArm],
        solution_instance: SolutionInstance,
        max_iterations: int,
        initial_temperature: float = 5.0,
        cooling_rate: float = 0.99,
        rho: float = 0.2,
        min_weight: float = 0.05,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.arms = arms
        self.max_iterations = max_iterations

        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate

        self.rho = rho
        self.min_weight = min_weight

        # Current / best state
        self.current_solution = solution_instance
        self.current_solution.rebuild_conflicts_and_violations()
        self.current_cost = self.current_solution.violation_count

        self.best_solution = self.current_solution.copy()
        self.best_cost = self.current_cost

    def select_arm(self) -> ALNSArm:
        """Roulette-wheel selection proportional to arm.weight."""
        total = sum(a.weight for a in self.arms)
        if total <= 0:
            return random.choice(self.arms)

        r = random.random() * total
        acc = 0.0
        for arm in self.arms:
            acc += arm.weight
            if acc >= r:
                return arm
        return self.arms[-1]

    def accept(self, old_cost: int, new_cost: int) -> bool:
        """Simulated annealing acceptance."""
        if new_cost <= old_cost:
            return True
        if self.temperature <= 1e-9:
            return False

        delta = new_cost - old_cost
        prob = math.exp(-delta / self.temperature)
        return random.random() < prob

    def compute_reward(self, old_cost: int, new_cost: int, accepted: bool) -> float:
        """Simple reward shaping for operator weights."""
        if new_cost < self.best_cost:
            return 5.0
        if accepted and new_cost < old_cost:
            return 3.0
        if accepted:
            return 1.0
        return 0.0

    def update_arm(self, arm: ALNSArm, reward: float) -> None:
        """Exponential smoothing update for roulette weights."""
        arm.times_used += 1
        arm.total_reward += reward
        arm.weight = max(self.min_weight, (1.0 - self.rho) * arm.weight + self.rho * reward)

    def run(self, verbose: bool = False) -> SolutionInstance:
        """Run the ALNS loop and return the best solution found."""
        for it in range(1, self.max_iterations + 1):
            arm = self.select_arm()

            candidate = self.current_solution.copy()

            # Destroy step: operator takes schedule, returns partial schedule
            partial_schedule = arm.destroy_operator(candidate.schedule)
            
            # Repair step: use ALNS repair method
            repaired_schedule = candidate.alns.repair(partial_schedule)
            
            # Repair failure handling
            if repaired_schedule is None:
                self.update_arm(arm, 0.0)
                continue
            
            # Update candidate with repaired schedule
            candidate.schedule = repaired_schedule
            candidate.rebuild_conflicts_and_violations()
            candidate_cost = candidate.violation_count

            accepted = self.accept(self.current_cost, candidate_cost)
            old_cost = self.current_cost

            if accepted:
                self.current_solution = candidate
                self.current_cost = candidate_cost

            if candidate_cost < self.best_cost:
                self.best_solution = candidate.copy()
                self.best_cost = candidate_cost

            reward = self.compute_reward(old_cost, candidate_cost, accepted)
            self.update_arm(arm, reward)

            self.temperature *= self.cooling_rate

            if verbose and it % 10 == 0:
                print(f"[{it:4d}] best={self.best_cost} current={self.current_cost} T={self.temperature:.4f}")

        return self.best_solution
