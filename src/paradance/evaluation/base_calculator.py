from abc import ABCMeta, abstractmethod
from functools import partialmethod
from typing import List, Union

import pandas as pd

from .auc_triple_parameters_evaluator import calculate_auc_triple_parameters
from .corrcoef_evaluator import calculate_corrcoef
from .distinct_portfolio_evaluator import (
    calculate_distinct_count_portfolio_concentration,
)
from .inverse_pair_evaluator import calculate_inverse_pair
from .log_mse_evaluator import calculate_log_mse
from .neg_rank_ratio_evaluator import calculate_neg_rank_ratio
from .portfolio_evaluator import calculate_portfolio_concentration
from .tau_evaluator import calculate_tau
from .top_coverage_evaluator import (
    calculate_distinct_top_coverage,
    calculate_top_coverage,
)
from .woauc_evaluator import calculate_woauc
from .wuauc_evaluator import calculate_wuauc


class BaseCalculator(metaclass=ABCMeta):
    """
    A base class for calculators that provides partial methods to calculate different
    evaluation metrics.

    This class is designed to be subclassed by specific calculator implementations
    that compute an overall score based on a combination of the metrics.
    """

    calculate_auc_triple_parameters = partialmethod(calculate_auc_triple_parameters)
    calculate_corrcoef = partialmethod(calculate_corrcoef)
    calculate_distinct_count_portfolio_concentration = partialmethod(
        calculate_distinct_count_portfolio_concentration
    )
    calculate_distinct_top_coverage = partialmethod(calculate_distinct_top_coverage)
    calculate_inverse_pair = partialmethod(calculate_inverse_pair)
    calculate_log_mse = partialmethod(calculate_log_mse)
    calculate_neg_rank_ratio = partialmethod(calculate_neg_rank_ratio)
    calculate_portfolio_concentration = partialmethod(calculate_portfolio_concentration)
    calculate_tau = partialmethod(calculate_tau)
    calculate_top_coverage = partialmethod(calculate_top_coverage)
    calculate_woauc = partialmethod(calculate_woauc)
    calculate_wuauc = partialmethod(calculate_wuauc)

    def __init__(self, selected_columns: List[str]) -> None:
        """Initializes the BaseCalculator."""
        self.selected_columns = selected_columns
        self.evaluated_dataframe: pd.DataFrame = pd.DataFrame()
        self.samplers: dict = {}
        self.woauc_dict: dict = {}
        self.bin_mappings: dict = {}

    @abstractmethod
    def get_overall_score(self, weights_for_equation: List[float]) -> None:
        """
        Calculates the overall score based on the weights provided for each evaluation metric.
        """
        pass

    def clip_max(
        self, que_a: Union[pd.Series, float, int], que_b: Union[pd.Series, float, int]
    ) -> pd.Series:
        print(type(que_a))
        if isinstance(que_a, pd.Series):
            return que_a.clip(lower=que_b)
        elif isinstance(que_b, pd.Series):
            return que_b.clip(lower=que_a)

    @staticmethod
    def clip_min(que_a: pd.Series, que_b: pd.Series) -> pd.Series:
        # return pd.Series(que_a).clip(upper=2)
        # return sum(pd.Series(que_a), 2)
        a = pd.Series(que_a).clip(upper=2)
        return pd.Series(a)
        # return que_a.clip(upper=que_b)
        # if isinstance(que_a, pd.Series):
        #     return que_a.clip(upper=que_b)
        # elif isinstance(que_b, pd.Series):
        #     return que_b.clip(upper=que_a)

    def initialize_local_dict(
        self, weights_for_equation: List[float], columns: List
    ) -> dict:
        """
        Initializes a dictionary that can be used for additional calculations.
        """
        local_dict = {
            "weights": weights_for_equation,
            "columns": columns,
            "sum": sum,
            "max": self.clip_max,
            "min": self.clip_min,
        }
        return local_dict
