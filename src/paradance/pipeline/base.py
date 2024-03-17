import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Union

from ..dataloader import CSVLoader, ExcelLoader, load_config
from ..evaluation import Calculator, LogarithmPCACalculator
from ..optimization import MultipleObjective, optimize_run

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
        self.file_type = self.config["DataLoader"].get("file_type", "csv")
        self.n_trials = n_trials

    def _load_dataset(self) -> None:
        """Loads the dataset based on the file type specified in the configuration.

        Supports loading from CSV and Excel files.
        """
        if self.file_type == "csv":
            self.dataframe = CSVLoader(config=self.config["DataLoader"]).df
        elif self.file_type == "xlsx":
            self.dataframe = ExcelLoader(config=self.config["DataLoader"]).df

    @abstractmethod
    def _load_calculator(self) -> Union[Calculator, LogarithmPCACalculator]:
        """Load or define the calculator for the pipeline operations.

        This method should be implemented to load or define the calculator, which
        might be a model or any computational tool needed for the pipeline.
        """
        pass

    def _add_objective(
        self, calculator: Union[Calculator, LogarithmPCACalculator]
    ) -> None:
        """Defines the optimization objective for PCA."""
        self.objective = MultipleObjective(
            calculator=calculator,
            config=self.config["Objective"],
        )

    def _add_evaluators(self) -> None:
        """Adds evaluators for optimization based on configuration settings."""
        flags = self.config["Evaluator"]["flags"]
        labels = self.config["Evaluator"]["labels"]
        for flag, label in zip(flags, labels):
            self.objective.add_evaluator(
                flag=flag,
                target_column=label,
            )

    def _optimize(self) -> None:
        """Runs the optimization process for the defined objective and evaluators."""
        optimize_run(
            multiple_objective=self.objective,
            n_trials=self.n_trials,
        )

    @abstractmethod
    def show_results(self) -> None:
        """Displays the results of the optimization process."""
        pass

    def run(self) -> None:
        """Execute the defined pipeline operations.

        This method orchestrates the execution of the pipeline by calling the
        abstract methods in sequence. It logs the beginning of the process and
        ensures that each step is performed in the correct order.
        """
        logger.info("Running pipeline...")
        self._load_dataset()
        calculator = self._load_calculator()
        self._add_objective(calculator)
        self._add_evaluators()
        self._optimize()
