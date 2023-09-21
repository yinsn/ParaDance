import subprocess
from typing import Union

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

    save_study(ob)
    if not ob.save_study:
        subprocess.run(["rm", "-rf", ob.full_path])
