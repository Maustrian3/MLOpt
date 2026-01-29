import math
import random

from meta_heuristics.MAB.alns_arm import ALNSArm
from meta_heuristics.adaptive_large_neighborhood_search import ALNS
from meta_heuristics.problem_instance import ProblemInstance
from meta_heuristics.solution_instance import SolutionInstance


class ALNSMultiArmedBandit:
    def __init__(
        self,
        arms: list[ALNSArm],
        solution_instance: SolutionInstance,
        max_iterations: int,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.995,
        rho: float = 0.2,
        min_weight: float = 0.05,
        seed: int | None = None,
    ):
        """
        arms                : list of destroy+repair operator pairs
        solution_inst       : Solution instance
        max_iterations      : ALNS iterations
        initial_temperature : SA starting temperature
        cooling_rate        : SA cooling factor
        rho                 : ALNS learning rate
        min_weight          : lower bound on operator weights
        """

        if seed is not None:
            random.seed(seed)

        self.arms = arms
        self.solution_inst = solution_instance
        self.max_iterations = max_iterations

        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate

        self.rho = rho
        self.min_weight = min_weight

        self.best_cost = self.solution_inst.violation_count

    def select_arm(self):
        """
        Roulette-wheel selection
        """
        # Collect the current weights of all arms.
        weights = [arm.weight for arm in self.arms]

        # Total weight is used to normalize probabilities implicitly.
        total = sum(weights)

        # Safety fallback: if all weights are zero, choose random
        if total == 0:
            return random.choice(self.arms)

        r = random.random() * total

        # Choose arm
        acc = 0.0
        for arm in self.arms:
            acc += arm.weight

            # Once the cumulative weight exceeds r,
            # we select this arm.
            # Probability of selecting arm i is:
            #   arm.weight / total
            if acc >= r:
                return arm

        # Fallback due to rounding errors: select last arm
        return self.arms[-1]

    def accept(self, old_cost: float, new_cost: float) -> bool:
        """
        Simulated annealing acceptance.
        """
        if new_cost <= old_cost:
            return True

        delta = new_cost - old_cost
        prob = math.exp(-delta / self.temperature)

        return random.random() < prob

    def compute_reward(
        self,
        old_cost: float,
        new_cost: float,
        accepted: bool,
    ) -> float:
        """
        Reward scheme
        """
        if new_cost < self.best_cost:
            return 5.0
        elif accepted and new_cost < old_cost:
            return 3.0
        elif accepted:
            return 1.0
        else:
            return 0.0

    def update_arm(self, arm: ALNSArm, reward: float):
        """
        Exponential smoothing update of operator weight.
        """
        arm.times_used += 1
        arm.total_reward += reward

        arm.weight = max(
            self.min_weight,
            (1 - self.rho) * arm.weight + self.rho * reward
        )

    def run(self, verbose: bool = False):
        """
        Run ALNS for max_iterations.
        Returns best solution found.
        """

        for it in range(1, self.max_iterations + 1):
            arm = self.select_arm()

            candidate = self.solution_inst.copy()

            arm.destroy(candidate)
            candidate.repair()

            candidate_cost = candidate.count_violations()

            # Acceptance decision
            accepted = self.accept(self.current_cost, candidate_cost)

            old_cost = self.current_cost

            if accepted:
                self.current_solution = candidate
                self.current_cost = candidate_cost

                if candidate_cost < self.best_cost:
                    self.best_solution = candidate
                    self.best_cost = candidate_cost

            # Reward operator
            reward = self.compute_reward(
                old_cost,
                candidate_cost,
                accepted
            )
            self.update_arm(arm, reward)

            # Cool temperature
            self.temperature *= self.cooling_rate

            if verbose and it % 10 == 0:
                print(
                    f"[{it:5d}] "
                    f"best={self.best_cost:.2f} "
                    f"current={self.current_cost:.2f} "
                    f"T={self.temperature:.4f}"
                )

        return self.best_solution

if __name__ == '__main__':
    arms = [
        ALNSArm("destroy_weeks_2", SolutionInstance.destroy_weeks_2),
        ALNSArm("destroy_groups_03", SolutionInstance.destroy_groups_03),
        ALNSArm("destroy_pairs_2", SolutionInstance.destroy_player_pairs_2),
    ]

    problem_instance = ProblemInstance(3,10,4)
    solution_inst = SolutionInstance(problem_instance, SolutionInstance.generate_random_solution)
    print(solution_inst)

    alns = ALNSMultiArmedBandit(
        arms=arms,
        solution_instance=solution_inst,
        max_iterations=500,
        initial_temperature=5.0,
        cooling_rate=0.99,
    )

    best = alns.run(verbose=True)

    # TODO debug and test if works