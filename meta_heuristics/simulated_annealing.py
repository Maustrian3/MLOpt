import math
import numpy as np

from meta_information import MetaInformation
from neighborhood import Neighborhood
from problem_instance import ProblemInstance
from solution_instance import SolutionInstance
from swap_2_player import Swap2Player


class SimulatedAnnealing:
    """Implements the simulated annealing metaheuristic with reheating to solve the social golfer problem."""

    def __init__(self,
                 solution_inst: SolutionInstance,
                 init_temperature: float,
                 final_temperature: float,
                 equilibrium_iterations: int,
                 alpha: float,
                 max_reheats: int = 10,
                 reheat_factor: float = 0.5,
                 no_improvement_threshold: int = 5):
        """
        Args:
            solution_inst: Initial solution instance
            init_temperature: Starting temperature
            final_temperature: Minimum temperature before cooling cycle ends
            equilibrium_iterations: Number of iterations at each temperature
            alpha: Cooling rate (0 < alpha < 1)
            max_reheats: Maximum number of reheating cycles allowed
            reheat_factor: Factor to multiply with init_temperature for reheating (0 < factor <= 1)
            no_improvement_threshold: Number of cooling cycles without improvement before reheating
        """
        self.solution_instance = solution_inst
        self.best_solution = solution_inst.copy() if hasattr(solution_inst, 'copy') else solution_inst
        self.best_violations = solution_inst.violation_count

        self.init_temperature = init_temperature
        self.temperature = init_temperature
        self.final_temp = final_temperature
        self.equi_iter = equilibrium_iterations
        self.alpha = alpha

        self.max_reheats = max_reheats
        self.reheat_factor = reheat_factor
        self.no_improvement_threshold = no_improvement_threshold

        self.reheat_count = 0
        self.cycles_without_improvement = 0

        self.meta_inf = MetaInformation()

    def metropolis_criterion(self, solution_instance: SolutionInstance, delta_obj) -> bool:
        if solution_instance.is_better_obj(0, delta_obj):
            return True
        return np.random.random_sample() <= math.exp(-abs(delta_obj) / self.temperature)

    def cool_down(self):
        self.temperature *= self.alpha

    def reheat(self):
        """Reheat the system to escape local optima."""
        self.reheat_count += 1
        self.temperature = self.init_temperature * (self.reheat_factor ** self.reheat_count)
        self.cycles_without_improvement = 0
        print(f"\n{'=' * 60}")
        print(f"REHEATING #{self.reheat_count}: Temperature reset to {self.temperature:.4f}")
        print(f"Best violations so far: {self.best_violations}")
        print(f"{'=' * 60}\n")

    def solve(self):
        """Run SA with reheating until violations reach 0 or max reheats exceeded."""
        iteration = 0
        cycle = 0

        print(f"Starting SA with reheating (max {self.max_reheats} reheats)")
        print(f"Initial violations: {self.solution_instance.violation_count}\n")

        while self.solution_instance.violation_count > 0:
            cycle_start_violations = self.best_violations

            # Standard SA cooling cycle
            while self.temperature > self.final_temp:
                for i in range(self.equi_iter):
                    delta = self.solution_instance.generate_random_neighborhood_move()
                    acceptance = self.metropolis_criterion(self.solution_instance, delta)

                    if acceptance:
                        self.solution_instance.apply_neighborhood_move(delta)

                        # Track best solution found
                        if self.solution_instance.violation_count < self.best_violations:
                            self.best_violations = self.solution_instance.violation_count
                            # Store best solution if copy method exists
                            if hasattr(self.solution_instance, 'copy'):
                                self.best_solution = self.solution_instance.copy()

                        print(
                            f"Iter {iteration}: Accepted (delta={delta:+d}, violations={self.solution_instance.violation_count})")
                    else:
                        print(
                            f"Iter {iteration}: Rejected (delta={delta:+d}, violations={self.solution_instance.violation_count})")

                    iteration += 1
                    self.meta_inf.inc_iterations()

                    # Early exit if solution found
                    if self.solution_instance.violation_count == 0:
                        print(f"\n🎉 SOLUTION FOUND! Zero violations achieved at iteration {iteration}")
                        self.meta_inf.set_final_obj(0)
                        return self.solution_instance

                self.cool_down()
                self.meta_inf.add_iteration_data(self.solution_instance.violation_count)
                print(f"Cooled to T={self.temperature:.4f}, Best violations: {self.best_violations}\n")

            # Check if we improved in this cycle
            cycle += 1
            if self.best_violations >= cycle_start_violations:
                self.cycles_without_improvement += 1
            else:
                self.cycles_without_improvement = 0

            print(f"Cycle {cycle} complete. Cycles without improvement: {self.cycles_without_improvement}")

            # Decide whether to reheat
            if self.solution_instance.violation_count > 0:
                if self.reheat_count >= self.max_reheats:
                    print(f"\n⚠️  Maximum reheats ({self.max_reheats}) reached.")
                    print(f"Best solution found has {self.best_violations} violations")
                    break

                if self.cycles_without_improvement >= self.no_improvement_threshold:
                    self.reheat()
                else:
                    # Standard reheat if not improved enough
                    self.temperature = self.init_temperature * (self.reheat_factor ** (self.reheat_count + 1))
                    self.reheat_count += 1
                    print(f"\nReheating to T={self.temperature:.4f}")

        self.meta_inf.set_final_obj(self.best_violations)

        # Return best solution found
        if hasattr(self, 'best_solution') and self.best_violations < self.solution_instance.violation_count:
            return self.best_solution
        return self.solution_instance


if __name__ == "__main__":
    problem_instance = ProblemInstance(6, 32, 8)
    neighborhood = Swap2Player()
    solution_instance = SolutionInstance(
        problem_instance,
        SolutionInstance.generate_random_solution,
        neighborhood
    )

    print(f"Initial solution:\n{solution_instance}\n")

    sa_solver = SimulatedAnnealing(
        solution_instance,
        init_temperature=1000,
        final_temperature=0.01,
        equilibrium_iterations=20,
        alpha=0.90,
        max_reheats=10,
        reheat_factor=0.5,
        no_improvement_threshold=3
    )

    best_solution = sa_solver.solve()

    print(f"\n{'=' * 60}")
    print(f"FINAL RESULT")
    print(f"{'=' * 60}")
    print(f"Violations: {best_solution.count_violations()}")
    print(f"Total iterations: {sa_solver.meta_inf.iterations}")
    print(f"Reheats used: {sa_solver.reheat_count}")
    print(f"\nSolution:\n{best_solution}")