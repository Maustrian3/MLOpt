# Machine Learning for Optimization - Assignment 1

In this assignment, we solve the Social Golfer Problem with two approaches: an exact method (OR-Tools CP-SAT) and a
meta-heuristic method (Simulated Annealing). We then generate a classification dataset by running both solvers under the
same time limit and labeling each instance by which solver is faster. Finally, we train a Random Forest model as an
algorithm selector using at least five instance features and evaluate it with the required 65\%/35\% train/test split.
The evaluation includes accuracy, a confusion matrix, and a comparison against always choosing one solver.

## Social Golfer Problem Solver

The Social Golfer Problem can be stated as follows: How can P golfers play together in G groups once a week, for as many
weeks as possible, without two golfers meeting more than once? In this task, it is reformulated to already predefine the
weeks to check if there is a possible solution.

## Exact Method: Constraint Programming

For the exact method, we treat the task as a feasibility problem. If CP-SAT finds a schedule that satisfies all
constraints within the time limit, we count the instance as solved. We used OR-Tools as the solver framework.

- `constraint_programming\exact_method.py`: Implementation of the constraint-programming solution.
- `verification\verfiy.py`: Verification method to verify if found solution is correct.

## Meta-Heuristic Algorithm: Simulated Annealing

For the meta heuristic approach we used simulated annealing. We start with a random assignment which already fulfills
part of the constraints and use simulated annealing with a simple swap of 2 golfers as the neighborhood.

- `meta_heuristics`
    - `meta_info_plots.ipynb`: Conflicts per Iteration
    - `meta_information.py`: Helper class to collect metadata of runs
    - `neighborhood.py`: Defines the interface for neighborhood structures that generate and apply moves.
    - `problem_instance.py`: Defines the problem parameters and constraints
    - `simulated_annealing.py`: Implementation of the simulated annealing algorithm
    - `solution_instance.py`: Represents a complete solution/schedule and manages the conflict tracking.
    - `swap_2_player.py` Implementation of the swap 2 golfer neighborhood

## Algorithmic Selection & Evaluation

- `generated_dataset.py`: Runs both algorithms in the limited timeframe and collects metadata to build dataset for
  machine learning.
- `generate_test_instances.py`: Generates test instances of the social golfer problem
- `algorithm-comparison.ipynb` & `Notebook_Report`: Train and run machine learning classification algorithm and generate
  graphs for the report.

