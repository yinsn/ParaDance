import logging
from typing import Optional, Union

from mixician import SelfBalancingLogarithmPCACalculator

from ..evaluation import Calculator, LogarithmPCACalculator
from .base import BasePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class LogarithmPCAPipeline(BasePipeline):
    """Pipeline for processing and optimizing PCA with logarithmic transformations.

    This pipeline extends the `BasePipeline` class to implement a specific process for
    optimizing Principal Component Analysis (PCA) with logarithmic transformations,
    particularly focusing on self-balancing mechanisms.

    Attributes:
        file_type (str): Type of the file to load data from, supported types are 'csv' and 'xlsx'.
        dataframe (pd.DataFrame): The loaded dataset in a pandas DataFrame.
        calculator (LogarithmPCACalculator): Calculator for PCA operations.
        objective (MultipleObjective): The optimization objective.
    """

    def __init__(self, config_path: Optional[str] = None, n_trials: int = 200) -> None:
        """Initializes the pipeline with configuration and trial settings.

        Args:
            config_path (Optional[str]): Path to the configuration file. Defaults to None.
            n_trials (int): Number of optimization trials to perform. Defaults to 200.
        """
        super().__init__(
            config_path,
            n_trials,
        )
        self.run()

    def _load_calculator(self) -> Union[Calculator, LogarithmPCACalculator]:
        """Initializes the PCA calculator with the loaded dataset."""
        pca_calculator = SelfBalancingLogarithmPCACalculator(
            dataframe=self.dataframe,
            config=self.config["Calculator"],
        )
        self.calculator = LogarithmPCACalculator(
            pca_calculator=pca_calculator,
        )
        return self.calculator

    def show_results(self) -> None:
        """Displays the results of the optimization process."""
        self.calculator.pca_calculator.update(
            pca_weights=self.objective.best_params,
        )
        logger.info("Best parameters for PCA with logarithmic transformations:")
        self.calculator.pca_calculator.show_weights()
        self.calculator.pca_calculator.show_equation()
