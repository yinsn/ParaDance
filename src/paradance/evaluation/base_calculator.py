from abc import ABCMeta, abstractmethod
from functools import partialmethod
from typing import List

from .auc_triple_parameters_evaluator import calculate_auc_triple_parameters
from .distinct_portfolio_evaluator import (
    calculate_distinct_count_portfolio_concentration,
)
from .inverse_pair_evaluator import calculate_inverse_pair
from .log_mse_evaluator import calculate_log_mse
from .neg_rank_ratio_evaluator import calculate_neg_rank_ratio
from .portfolio_evaluator import calculate_portfolio_concentration
from .tau_evaluator import calculate_tau
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
    calculate_distinct_count_portfolio_concentration = partialmethod(
        calculate_distinct_count_portfolio_concentration
    )
    calculate_inverse_pair = partialmethod(calculate_inverse_pair)
    calculate_log_mse = partialmethod(calculate_log_mse)
    calculate_neg_rank_ratio = partialmethod(calculate_neg_rank_ratio)
    calculate_portfolio_concentration = partialmethod(calculate_portfolio_concentration)
    calculate_tau = partialmethod(calculate_tau)
    calculate_woauc = partialmethod(calculate_woauc)
    calculate_wuauc = partialmethod(calculate_wuauc)

    def __init__(self, selected_columns: List[str]) -> None:
        """Initializes the BaseCalculator."""
        self.selected_columns = selected_columns

    @abstractmethod
    def get_overall_score(self, weights_for_equation: List[float]) -> None:
        """
        Calculates the overall score based on the weights provided for each evaluation metric.
        """
        pass
