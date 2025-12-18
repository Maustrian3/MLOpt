import csv
import math
import time
import multiprocessing as mp
from pathlib import Path
from sys import exception

from meta_heuristics.simulated_annealing import SimulatedAnnealing
from meta_heuristics.problem_instance import ProblemInstance
from meta_heuristics.solution_instance import SolutionInstance
from meta_heuristics.swap_2_player import Swap2Player
from constraint_programming.exact_method import solve_social_golfer

# Config
INPUT_FILE = "SGP_test_instances.csv"
OUTPUT_FILE = "SGP_dataset.csv"

SA_TIME_LIMIT = 100.0      # seconds for full SA run
EXACT_TIME_LIMIT = 100.0   # seconds for OR-Tools CP-SAT
PROBE_TIME = 3     # time (s) for probing run



def compute_probe_conflicts(num_golfers: int,
                           num_weeks: int,
                           num_groups: int,
                           probe_time: float = PROBE_TIME) -> float:

    solver = solve_social_golfer(num_golfers=num_golfers,
                                 num_groups=num_groups,
                                 num_weeks=num_weeks,
                                 presolve=False,
                                 return_solver=True,
                                 max_time=probe_time)

    return solver.ResponseProto().num_integer_propagations



def _sa_worker(num_golfers, num_weeks, num_groups, queue):
    """Worker process for running SA to completion (or until parameters stop)."""

    problem = ProblemInstance(num_groups, num_golfers, num_weeks)

    neighborhood = Swap2Player()
    solution = SolutionInstance(problem, SolutionInstance.generate_random_solution, neighborhood)

    sa_solver = SimulatedAnnealing(
        solution_inst=solution,
        init_temperature=1000.0,
        final_temperature=0.01,
        equilibrium_iterations=20,
        alpha=0.90,
    )

    start = time.time()
    best_solution = sa_solver.solve()
    runtime = time.time() - start

    # If no feasible solution is found, throw exception to count as no solution found
    if best_solution.count_violations() != 0:
        raise Exception("Violation count != 0")

    # Only send back runtime
    queue.put(runtime)


def run_sa_with_timeout(num_golfers, num_weeks, num_groups,
                        time_limit: float = SA_TIME_LIMIT):
    """
    Runs SA in a separate process and enforces a wall-clock timeout.

    Returns:
        (finished: bool, runtime: float)
        runtime == time_limit if timeout occurred
    """
    q = mp.Queue()
    p = mp.Process(target=_sa_worker, args=(num_golfers, num_weeks, num_groups, q))
    p.start()
    p.join(timeout=time_limit)

    if p.is_alive():
        p.terminate()
        p.join()
        return False, time_limit

    try:
        runtime = q.get_nowait()
    except Exception:
        # In case of crash or no result
        runtime = time_limit
        return False, runtime

    return True, runtime


def _exact_worker(num_golfers, num_weeks, num_groups, time_limit, queue):
    """
    Worker process for running the OR-Tools exact method.
    """
    start = time.time()
    result = solve_social_golfer(num_golfers=num_golfers,
                            num_groups=num_groups,
                            num_weeks=num_weeks,
                             max_time=time_limit)
    runtime = time.time() - start


    # If no valid solution is found, throw exception to count as no solution found
    if result is None:
        raise Exception("No valid solution found.")

    # Only send back runtime
    queue.put(runtime)


def run_exact_with_timeout(num_golfers, num_weeks, num_groups,
                           time_limit: float = EXACT_TIME_LIMIT):
    """
    Runs the exact CP-SAT solver in a separate process with a wall-clock timeout.

    Returns:
        (finished: bool, runtime: float)
        runtime == time_limit if timeout occurred
    """
    q = mp.Queue()
    p = mp.Process(target=_exact_worker, args=(num_golfers, num_weeks, num_groups, time_limit, q))
    p.start()
    p.join(timeout=time_limit)

    if p.is_alive():
        p.terminate()
        p.join()
        return False, time_limit

    try:
        runtime = q.get_nowait()
    except Exception:
        runtime = time_limit
        return False, runtime

    return True, runtime


def decide_winner(sa_finished: bool, sa_runtime: float,
                  exact_finished: bool, exact_runtime: float) -> str:
    """
    Classification label:
        - 'sa'       if SA finished and exact didn't, or SA was faster
        - 'exact'    if exact finished and SA didn't, or exact was faster
        - 'both'     if both finished and times are equal (within small epsilon)
        - 'none'     if both timed out / failed
    """
    eps = 1e-6

    if not sa_finished and not exact_finished:
        return "none"
    if sa_finished and not exact_finished:
        return "sa"
    if exact_finished and not sa_finished:
        return "exact"

    # both finished
    if abs(sa_runtime - exact_runtime) <= eps:
        return "both"
    return "sa" if sa_runtime < exact_runtime else "exact"


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", newline="") as f_in, \
            output_path.open("w", newline="") as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = [
            "num_golfers",
            "num_weeks",
            "num_groups",
            "group_size",
            "tightness_T",
            "probe_conflicts",
            "sa_runtime",
            "exact_runtime",
            "sa_finished",
            "exact_finished",
            "winner",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            num_golfers = int(row["num_golfers"])
            num_weeks = int(row["num_weeks"])
            num_groups = int(row["num_groups"])

            group_size = num_golfers // num_groups

            # 1–3 are just raw instance parameters
            # 4. Tightness index T
            tightness_T = (num_weeks * (group_size - 1)) / (num_golfers - 1)

            # 5. Local search gradient probe (x iterations)
            try:
                probe_conflicts = compute_probe_conflicts(
                    num_golfers=num_golfers,
                    num_weeks=num_weeks,
                    num_groups=num_groups,
                    probe_time=PROBE_TIME,
                )
            except Exception as e:
                print(f"[WARN] LS probe failed for instance {row}: {e}")
                probe_conflicts = 0.0

            # Run SA with timeout
            sa_finished, sa_runtime = run_sa_with_timeout(
                num_golfers=num_golfers,
                num_weeks=num_weeks,
                num_groups=num_groups,
                time_limit=SA_TIME_LIMIT,
            )

            # Run exact method with timeout
            exact_finished, exact_runtime = run_exact_with_timeout(
                num_golfers=num_golfers,
                num_weeks=num_weeks,
                num_groups=num_groups,
                time_limit=EXACT_TIME_LIMIT,
            )

            # Decide winner
            winner = decide_winner(
                sa_finished, sa_runtime,
                exact_finished, exact_runtime
            )

            out_row = {
                "num_golfers": num_golfers,
                "num_weeks": num_weeks,
                "num_groups": num_groups,
                "group_size": group_size,
                "tightness_T": tightness_T,
                "probe_conflicts": probe_conflicts,
                "sa_runtime": sa_runtime,
                "exact_runtime": exact_runtime,
                "sa_finished": int(sa_finished),
                "exact_finished": int(exact_finished),
                "winner": winner,
            }
            writer.writerow(out_row)
            print(f"Processed instance: {out_row}")

    print(f"\nDataset written to {output_path}")


if __name__ == "__main__":
    main()
