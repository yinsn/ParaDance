import logging
from typing import Optional

import pandas as pd
from mixician import SelfBalancingLogarithmPCACalculator

from ..dataloader import CSVLoader, ExcelLoader
from ..evaluation import LogarithmPCACalculator
from ..optimization import MultipleObjective, optimize_run
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
        super().__init__(config_path, n_trials)
        self.file_type = self.config["DataLoader"].get("file_type", "csv")
        self.run()

    def _load_dataset(self) -> None:
        """Loads the dataset based on the file type specified in the configuration.

        Supports loading from CSV and Excel files.
        """
        if self.file_type == "csv":
            self.dataframe = CSVLoader(config=self.config["DataLoader"]).df
        elif self.file_type == "xlsx":
            self.dataframe = ExcelLoader(config=self.config["DataLoader"]).df

    def _load_calculator(self) -> None:
        """Initializes the PCA calculator with the loaded dataset."""
        pca_calculator = SelfBalancingLogarithmPCACalculator(
            dataframe=self.dataframe,
            config=self.config["Calculator"],
        )
        self.calculator = LogarithmPCACalculator(
            pca_calculator=pca_calculator,
        )

    def _add_objective(self) -> None:
        """Defines the optimization objective for PCA."""
        self.objective = MultipleObjective(
            calculator=self.calculator,
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

    def show_raw_data(self) -> pd.DataFrame:
        """Returns the raw dataset loaded from the file."""
        return self.calculator.pca_calculator.dataframe

    def show_results(self) -> None:
        """Displays the results of the optimization process.

        Updates the PCA calculator with the best parameters, shows the PCA weights.
        """
        logger.info(
            "Plotting logarithm distributions before Logarithm PCA optimization."
        )
        self.calculator.pca_calculator.plot_logarithm_distributions()
        self.calculator.pca_calculator.update(
            pca_weights=self.objective.best_params,
        )
        self.calculator.pca_calculator.plot_self_balancing_projected_distribution()
        logger.info("Best parameters for PCA with logarithmic transformations:")
        self.calculator.pca_calculator.show_weights()
        self.calculator.pca_calculator.show_equation()
