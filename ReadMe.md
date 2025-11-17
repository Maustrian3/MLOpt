# Social Golfer Problem Solver

A simulated annealing-based solver for the Social Golfer Problem, which aims to schedule golfers into groups over multiple weeks such that no two golfers play together more than once.

## Problem Description

Given:
- **N players** (golfers)
- **G groups** per week
- **W weeks**

Goal: Schedule players into groups each week such that no pair of players meets more than once.

## Project Structure

### Core Classes

#### 1. `ProblemInstance`
**Purpose:** Defines the problem parameters and constraints.

**Key Attributes:**
- `num_groups`: Number of groups per week
- `num_players`: Total number of players
- `num_weeks`: Number of weeks to schedule
- `groupsize`: Calculated size of each group (num_players / num_groups)

**Usage:**
```python
problem = ProblemInstance(num_groups=3, num_players=10, num_weeks=4)
```

---

#### 2. `SolutionInstance`
**Purpose:** Represents a complete solution/schedule and manages the conflict tracking.

**Key Attributes:**
- `problem_instance`: Reference to the problem definition
- `schedule`: 3D structure representing the solution
  - Format: `schedule[week][group][player_index]`
  - Example: `schedule[0][1][2]` = player in week 0, group 1, position 2
- `conflicts`: Symmetric NxN matrix tracking how many times each pair of players has met
  - `conflicts[i][j]` = number of times player i and player j have been in the same group
- `violation_count`: Total number of pairs that have met more than once
- `neighborhood`: The neighborhood structure used for generating moves

**Key Methods:**
- `generate_random_solution()`: Creates a random initial schedule
- `count_violations()`: Counts pairs that played together more than once
- `generate_random_neighborhood_move()`: Generates a random neighbor solution
- `apply_neighborhood_move(delta)`: Applies a move and updates the solution state
- `_update_conflicts_for_group(group, inc)`: Updates the conflict matrix when groups change

**Usage:**
```python
solution = SolutionInstance(
    problem_inst=problem,
    initial_solution_method=SolutionInstance.generate_random_solution,
    neighborhood=Swap2Player()
)
```

---

#### 3. `Neighborhood` (Abstract Base Class)
**Purpose:** Defines the interface for neighborhood structures that generate and apply moves.

**Abstract Methods:**
- `generate_random_neighborhood_move(problem_inst, schedule, conflicts)`: Generate a random move
- `apply_neighborhood_move(schedule, conflicts)`: Apply the move to the solution

**Concrete Implementation: `Swap2Player`**

---

#### 4. `Swap2Player`
**Purpose:** Implements a neighborhood structure that swaps two players from different groups.

**Move Strategy:**
1. Randomly select two weeks (can be the same week)
2. Randomly select a group in each week
3. Randomly select a player from each group
4. Swap the two players

**Key Attributes (stores move information):**
- `week_idx_a`, `week_idx_b`: Week indices for the swap
- `group_idx_a`, `group_idx_b`: Group indices within each week
- `player_idx_a`, `player_idx_b`: Player positions within each group
- `old_group_a`, `old_group_b`: Original group compositions

**Key Methods:**
- `_calculate_swap_delta()`: Efficiently calculates the change in violations without actually performing the swap
  - Returns: Change in violation count (negative = improvement)
  - Logic: Checks if conflicts increase (1→2) or decrease (2→1) for affected pairs
- `_update_conflicts()`: Updates the conflict matrix after a swap
- `apply_neighborhood_move()`: Performs the actual player swap in the schedule

---

#### 5. `SimulatedAnnealing`
**Purpose:** Implements the simulated annealing metaheuristic to solve the problem.

**Key Attributes:**
- `solution_instance`: Current solution being optimized
- `temperature`: Current temperature (controls acceptance of worse solutions)
- `final_temp`: Stopping temperature
- `equi_iter`: Number of iterations at each temperature (equilibrium iterations)
- `alpha`: Cooling rate (0 < alpha < 1)

**Key Methods:**
- `metropolis_criterion(solution_instance, delta_obj)`: Decides whether to accept a move
  - Always accepts improvements (delta < 0)
  - Accepts worse solutions with probability exp(-|delta| / temperature)
- `cool_down()`: Reduces temperature by factor alpha
- `solve()`: Main optimization loop

**Algorithm Flow:**
```
while temperature > final_temperature:
    for i in range(equilibrium_iterations):
        1. Generate random neighbor move
        2. Calculate delta (change in violations)
        3. Apply metropolis criterion
        4. If accepted: apply move
        5. Cool down temperature
```

**Usage:**
```python
sa_solver = SimulatedAnnealing(
    solution_inst=solution,
    init_temperature=1000,
    final_temperature=0.01,
    equilibrium_iterations=20,
    alpha=0.90
)
best_solution = sa_solver.solve()
```

---

## Class Interaction Diagram

```
ProblemInstance
       ↓
SolutionInstance ←→ Neighborhood (abstract)
       ↓                    ↓
SimulatedAnnealing    Swap2Player (concrete)
```

**Flow:**
1. **ProblemInstance** defines the problem parameters
2. **SolutionInstance** creates an initial solution and tracks conflicts
3. **Neighborhood** (via **Swap2Player**) generates and applies moves
4. **SimulatedAnnealing** orchestrates the optimization process

---

## Key Data Structures

### Schedule Structure
```python
schedule = [
    # Week 0
    [
        [player_0, player_1, player_2],  # Group 0
        [player_3, player_4, player_5],  # Group 1
        [player_6, player_7, player_8]   # Group 2
    ],
    # Week 1
    [...],
    ...
]
```

### Conflict Matrix
```python
conflicts = [
    [0, 1, 0, 2, ...],  # Player 0's meetings with others
    [1, 0, 1, 0, ...],  # Player 1's meetings with others
    ...
]
# conflicts[i][j] = number of times player i and j have been grouped together
# Violation occurs when conflicts[i][j] > 1
```

---