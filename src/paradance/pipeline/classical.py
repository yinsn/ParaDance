import logging
from typing import Optional, Union

from ..evaluation import Calculator, LogarithmPCACalculator
from ..pipeline import BasePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ClassicalPipeline(BasePipeline):
    """Implements a classical pipeline for calculations.

    This class extends `BasePipeline` to implement a pipeline specifically
    designed for classical calculations. It initializes the pipeline based
    on a configuration path and number of trials, loads a calculator based on
    the configuration, and shows the results of the calculations.
    """

    def __init__(self, config_path: Optional[str] = None, n_trials: int = 200) -> None:
        """Initializes the classical pipeline.

        Args:
            config_path: The path to the configuration file, optional.
            n_trials: The number of trials to run, defaults to 200.
        """
        super().__init__(
            config_path,
            n_trials,
        )
        self.run()

    def _load_calculator(self) -> Union[Calculator, LogarithmPCACalculator]:
        """Loads the calculator based on the configuration.

        Depending on the configuration, initializes and returns an appropriate
        calculator for the pipeline.

        Returns:
            An instance of `Calculator` or `LogarithmPCACalculator` as specified
            by the configuration.
        """
        config = self.config["Calculator"]
        self.calculator = Calculator(
            df=self.dataframe,
            selected_columns=config.get("selected_columns", None),
            equation_type=config.get("equation_type", "product"),
            weights_for_groups=config.get("weights_for_groups", None),
            equation_eval_str=config.get("equation_eval_str", None),
        )

        return self.calculator

    def show_results(self) -> None:
        """Displays the results of the calculation.

        Logs information about the selected columns, first order weights, and
        power weights based on the calculations performed.
        """
        best_params = list(self.objective.study.best_params.values())
        if not (self.objective.first_order):
            first_order_weights = None
            power_weights = best_params
        elif (
            self.objective.calculator.equation_type == "product"
            and self.objective.power
        ):
            first_order_weights = best_params[self.objective.weights_num :]
            power_weights = best_params[: self.objective.weights_num]
        else:
            first_order_weights = best_params
            power_weights = None

        logger.info(f"Selected columns: {self.calculator.selected_columns}")
        logger.info(f"First order weights: {first_order_weights}")
        logger.info(f"Power weights: {power_weights}")
