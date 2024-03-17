from typing import Optional, Union

from ..evaluation import Calculator, LogarithmPCACalculator
from ..pipeline import BasePipeline


class ClassicalPipeline(BasePipeline):
    def __init__(self, config_path: Optional[str] = None, n_trials: int = 200) -> None:
        super().__init__(
            config_path,
            n_trials,
        )
        self.run()

    def _load_calculator(self) -> Union[Calculator, LogarithmPCACalculator]:
        config = self.config["Calculator"]
        self.calculator = Calculator(
            df=self.dataframe,
            selected_columns=config.get("selected_columns", None),
            equation_type=config.get("equation_type", "product"),
            weights_for_groups=config.get("weights_for_groups", None),
            equation_eval_str=config.get("equation_eval_str", None),
        )

        return self.calculator
