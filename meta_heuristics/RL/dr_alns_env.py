from meta_heuristics.solution_instance import SolutionInstance

import torch

device = torch.device("cpu")
torch.set_num_threads(1)

class DRALNSEnv:
    def __init__(self, problem_instance, max_steps):
        self.problem = problem_instance
        self.max_steps = max_steps

        self.reset()

    def reset(self):
        self.solution = SolutionInstance(
            self.problem,
            SolutionInstance.generate_random_solution
        )
        self.solution.rebuild_conflicts_and_violations()

        self.best_solution = self.solution.copy()
        self.best_violations = self.solution.violation_count

        self.prev_violations = self.solution.violation_count
        self.stagnation = 0
        self.step_count = 0

        self.last_accepted = False
        self.last_improved = False
        self.last_best_improved = False

        return self._get_state()

    def step(self, action):
        """
        action: int = index of destroy operator
        """
        self.step_count += 1

        old_solution = self.solution.copy()
        old_violations = self.solution.violation_count

        # Apply destroy operator
        if action == 0:
            self.solution.destroy_weeks_2()
        elif action == 1:
            self.solution.destroy_groups_03()
        elif action == 2:
            self.solution.destroy_player_pairs_2()
        else:
            raise ValueError("Invalid action")

        self.solution.repair()
        self.solution.rebuild_conflicts_and_violations()

        new_violations = self.solution.violation_count

        # Acceptance (simple greedy)
        # Accepts only if the solution doesn’t get worse.
        if new_violations <= old_violations:
            accepted = True
        else:
            accepted = False
            self.solution = old_solution  # revert
            new_violations = old_violations

        # Improvement signals
        current_improved = accepted and (new_violations < old_violations)

        best_improved = False
        if new_violations < self.best_violations:
            self.best_solution = self.solution.copy()
            self.best_violations = new_violations
            best_improved = True
            self.stagnation = 0
        else:
            self.stagnation += 1

        # Reward (sparse reward)
        # high reward only if you improve the best solution, small reward if current solution improved.
        reward = 0.0
        if best_improved:
            reward = 5.0
        elif current_improved:
            reward = 1.0

        # Keep everything up to date
        self.prev_violations = new_violations
        self.last_accepted = accepted
        self.last_improved = current_improved
        self.last_best_improved = best_improved

        done = self.step_count >= self.max_steps

        return self._get_state(), reward, done

    def _get_state(self):
        if self.solution.violation_count <= self.best_violations:
            cost_diff_best = -1.0
        else:
            cost_diff_best = (
                (self.solution.violation_count - self.best_violations)
                / max(1, self.best_violations)
            )

        # State features:
        # 1. BestImproved            (0/1)
        # 2. CurrentAccepted         (0/1)
        # 3. CurrentImproved         (0/1)
        # 4. IsCurrentBest           (0/1)
        # 5. CostDifferenceBest      (float)
        # 6. StagnationCount         (int)
        # 7. SearchBudgetUsed        (float in [0,1])
        return torch.tensor([
            int(self.last_best_improved),
            int(self.last_accepted),
            int(self.last_improved),
            int(self.solution.violation_count == self.best_violations),
            cost_diff_best,
            float(self.stagnation),
            self.step_count / self.max_steps
        ], dtype=torch.float32, device=device)
