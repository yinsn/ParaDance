import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
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

    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        config_path: Optional[str] = None,
        n_trials: int = 200,
    ) -> None:
        """Initializes the pipeline with configuration and trial settings.

        Args:
            config_path (Optional[str]): Path to the configuration file. Defaults to None.
            n_trials (int): Number of optimization trials to perform. Defaults to 200.
        """
        super().__init__(
            dataframe=dataframe, config_path=config_path, n_trials=n_trials
        )
        self._pre_run()

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

    def plot_logarithm_distributions(self) -> None:
        """Plots the logarithmic distributions of the dataset."""
        self.calculator.pca_calculator.plot_logarithm_distributions()

    def plot_self_balancing_projected_distribution(
        self, pca_weights: np.ndarray
    ) -> None:
        """This method updates the PCA weights in the calculator's PCA component and then
        plots the distribution based on these updated weights.
        """
        self.calculator.pca_calculator.update_pca_weights(
            pca_weights=pca_weights,
        )
        self.calculator.pca_calculator.plot_self_balancing_projected_distribution()

    def show_results(self) -> None:
        """Displays the results of the optimization process."""
        self.calculator.pca_calculator.update(
            pca_weights=self.objective.best_params,
        )
        logger.info("Best parameters for PCA with logarithmic transformations:")
        self.calculator.pca_calculator.get_weights()
        self.objective.build_logger()
        self.objective.logger.info(
            f"Best parameters: {self.calculator.pca_calculator.results}"
        )
        self.calculator.pca_calculator.show_equation()
