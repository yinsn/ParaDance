import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

from ..dataloader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class BasePipeline(metaclass=ABCMeta):
    """Abstract base class for implementing processing pipelines.

    This class provides a structured way to define a sequence of operations
    for data processing and optimization tasks. It is designed to be subclassed
    with specific implementations of the abstract methods provided.

    Attributes:
        config (Dict): Configuration settings loaded from a configuration file.
        n_trials (int): The number of optimization trials to perform.

    """

    def __init__(self, config_path: Optional[str] = None, n_trials: int = 200) -> None:
        self.config: Dict = load_config(config_path)
        self.n_trials = n_trials

    @abstractmethod
    def _load_dataset(self) -> None:
        """Load the dataset required for the pipeline operations.

        This method should be implemented to load and possibly preprocess the dataset
        needed for further steps in the pipeline.
        """
        pass

    @abstractmethod
    def _load_calculator(self) -> None:
        """Load or define the calculator for the pipeline operations.

        This method should be implemented to load or define the calculator, which
        might be a model or any computational tool needed for the pipeline.
        """
        pass

    @abstractmethod
    def _add_objective(self) -> None:
        """Define the optimization objective.

        This method should be implemented to define the objective function that
        will be optimized in the pipeline.
        """
        pass

    @abstractmethod
    def _add_evaluators(self) -> None:
        """Add evaluators for the optimization process.

        This method should be implemented to add any evaluators or metrics that
        will be used to assess the performance of the optimization process.
        """
        pass

    @abstractmethod
    def _optimize(self) -> None:
        """Perform the optimization process.

        This method should be implemented to perform the actual optimization,
        using the objective and evaluators defined in previous steps.
        """
        pass

    @abstractmethod
    def show_results(self) -> None:
        """Display the results of the optimization process.

        This method should be implemented to present the results of the optimization
        in a meaningful way, which might include logging or visualization.
        """
        pass

    def run(self) -> None:
        """Execute the defined pipeline operations.

        This method orchestrates the execution of the pipeline by calling the
        abstract methods in sequence. It logs the beginning of the process and
        ensures that each step is performed in the correct order.
        """
        logger.info("Running pipeline...")
        self._load_dataset()
        self._load_calculator()
        self._add_objective()
        self._add_evaluators()
        self._optimize()
