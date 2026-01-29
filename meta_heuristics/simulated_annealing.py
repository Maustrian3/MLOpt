# import math
# import sys
#
# import numpy as np
#
# from meta_heuristics.meta_information import MetaInformation
# from meta_heuristics.neighborhood import Neighborhood
# from meta_heuristics.problem_instance import ProblemInstance
# from meta_heuristics.solution_instance import SolutionInstance
# from meta_heuristics.swap_2_player import Swap2Player
#
#
# class SimulatedAnnealing:
#     """Implements the simulated annealing metaheuristic to solve the social golfer problem."""
#
#     def __init__(self,
#                  solution_inst: SolutionInstance,
#                  init_temperature: float,
#                  final_temperature: float,
#                  equilibrium_iterations: int,
#                  alpha: float,
#                  max_iterations: int = sys.maxsize,
#                  log: bool = False):
#
#         self.solution_instance = solution_inst
#
#         self.temperature = init_temperature
#         self.final_temp = final_temperature
#         self.equi_iter = equilibrium_iterations
#         self.alpha = alpha
#
#         self.max_iter = max_iterations
#
#         self.meta_inf = MetaInformation()
#         self.logging = log
#
#     def metropolis_criterion(self, solution_instance: SolutionInstance, delta_obj) -> bool:
#         if solution_instance.is_better_obj(0, delta_obj):                            # Is solution objectively better?
#             return True                                                                         # Take new solution instantly
#         return np.random.random_sample() <= math.exp(-abs(delta_obj) / self.temperature)    # Otherwise: sometimes accept worse solution
#
#     def cool_down(self):
#         self.temperature *= self.alpha
#
#     def solve(self):
#         iteration = 0
#         while self.temperature > self.final_temp:
#             for i in range(self.equi_iter):
#                 delta = self.solution_instance.generate_random_neighborhood_move()
#                 acceptance = self.metropolis_criterion(self.solution_instance, delta)
#                 if acceptance:
#                     self.solution_instance.apply_neighborhood_move(delta)
#                     if self.logging:
#                         print(
#                             f"Iteration {iteration}: Accepted move with delta={delta}, violations={self.solution_instance.violation_count}")
#                 else:
#                     if self.logging:
#                         print(
#                             f"Iteration {iteration}: Rejected move with delta={delta}, violations={self.solution_instance.violation_count}")
#
#                 iteration += 1
#                 self.meta_inf.inc_iterations()
#
#                 if iteration > self.max_iter:
#                     return self.solution_instance
#
#             self.cool_down()
#             self.meta_inf.add_iteration_data(self.solution_instance.violation_count)
#             if self.logging:
#                 print(f"Cooled down to temperature: {self.temperature}\n")
#         self.meta_inf.set_final_obj(self.solution_instance.violation_count)
#         return self.solution_instance
#
#
# if __name__ == "__main__":
#     problem_instance = ProblemInstance(3,10,4)
#     neighborhood = Swap2Player()
#     solution_instance = SolutionInstance(problem_instance, SolutionInstance.generate_random_solution, neighborhood)
#     print(solution_instance)
#     sa_solver = SimulatedAnnealing(
#         solution_instance,
#         init_temperature=1000,
#         final_temperature=0.01,
#         equilibrium_iterations=20,
#         alpha=0.90,
#     )
#     best_solution = sa_solver.solve()
#     print(f"Best solution found with {best_solution.count_violations()} violations: \n{best_solution}")
