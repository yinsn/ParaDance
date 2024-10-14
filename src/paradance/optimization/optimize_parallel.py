import csv
import subprocess
import sys
import threading
import time
from typing import Union

import numpy as np
from joblib import Parallel, delayed

from .get_processors import get_logical_processors_count
from .multiple_objective import MultipleObjective
from .save_study import save_study


def parallel_optimize(
    multiple_objective: MultipleObjective, i: int, ntrials: int
) -> None:
    """
    Optimize a multiple objective instance for a certain number of trials.

    Args:
        multiple_objective (MultipleObjective): The multiple objective instance to be optimized.
        i (int): The identifier for this parallel run, useful for tasks that require unique identifiers or handling per run.
        ntrials (int): The number of trials for optimization.

    Returns:
        None
    """
    multiple_objective.optimize(ntrials)


def get_best_trials(
    multiple_objective: MultipleObjective, refresh_rate: int = 5
) -> None:
    """
    Extracts and saves the best trials from the provided log content.
    """
    ob = multiple_objective
    file_path = f"{ob.full_path}/paradance.log"
    output_path = f"{ob.full_path}/paradance_best_trials.csv"

    while True:
        try:
            best_trials = set()
            with open(file_path, "r") as file:
                lines = file.readlines()

            extracted_data = []
            for idx, line in enumerate(lines):
                if "Best is trial" in line:
                    trial_number = int(line.split("Best is trial")[1].split(" ")[1])
                    if trial_number in best_trials:
                        continue

                    best_trials.add(trial_number)

                    for sub_idx in range(idx, -1, -1):
                        if (
                            f"Trial {trial_number} finished with result:"
                            in lines[sub_idx]
                        ):
                            results_line = lines[sub_idx]
                            sub_idx += 1

                            while "targets:" not in lines[sub_idx]:
                                sub_idx += 1
                            targets_line = (
                                lines[sub_idx].split("targets:")[1].strip().strip("[]")
                            )

                            while "weights:" not in lines[sub_idx]:
                                sub_idx += 1
                            weights_line = (
                                lines[sub_idx].split("weights:")[1].strip().strip("[]")
                            )

                            results = float(results_line.split("result:")[1].strip())
                            targets_str = [
                                val.strip()
                                for val in targets_line.split(",")
                                if val.strip()
                            ]
                            weights_str = [
                                val.strip()
                                for val in weights_line.split(",")
                                if val.strip()
                            ]

                            try:
                                targets = [float(val) for val in targets_str]
                                weights = [float(val) for val in weights_str]
                                extracted_data.append(
                                    (results, trial_number, targets, weights)
                                )
                            except ValueError as e:
                                sys.stdout.write(
                                    f"Error processing line: {sub_idx}, error: {e}\n"
                                )

                    with open(output_path, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(
                            [
                                f"{ob.formula}",
                                "Trial",
                                f"{ob.evaluator_flags}",
                                f"{ob.calculator.selected_columns}",
                            ]
                        )
                        for data in extracted_data:
                            writer.writerow(
                                [data[0], data[1], str(data[2]), str(data[3])]
                            )
        except FileNotFoundError:
            pass
        time.sleep(refresh_rate)


def optimize_run(
    multiple_objective: MultipleObjective,
    n_trials: int,
    parallel: Union[bool, int] = True,
) -> None:
    """
    Optimize the multiple objective in parallel using specified number of processors or all available ones.

    Args:
        multiple_objective (MultipleObjective): The multiple objective instance to be optimized.
        n_trials (int): Total number of trials for optimization, distributed across cores.
        parallel (Union[bool, int]): If True, use all available cores. If False, don't use parallelism.
                                    If int, use the specified number of cores.

    Returns:
        None
    """
    ob = multiple_objective
    log_listener_thread = threading.Thread(
        target=get_best_trials, args=(multiple_objective,)
    )
    log_listener_thread.daemon = True
    log_listener_thread.start()

    if isinstance(parallel, bool) and not parallel:
        multiple_objective.optimize(n_trials)
    else:
        n_cores = (
            get_logical_processors_count() if isinstance(parallel, bool) else parallel
        )
        unit_n_trials = n_trials // n_cores

        Parallel(n_jobs=n_cores)(
            delayed(parallel_optimize)(ob, i, unit_n_trials) for i in range(n_cores)
        )
    ob.best_params = np.asarray(list(ob.study.best_params.values()))
    save_study(ob)
    if not ob.save_study:
        subprocess.run(["rm", "-rf", ob.full_path])
