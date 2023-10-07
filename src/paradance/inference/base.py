from abc import ABC, abstractmethod

import numpy as np
import pymc as pm


class BayesianBaseModel(ABC):
    """
    BayesianBaseModel Abstract Base Class.

    This class represents the basic structure and methods required for Bayesian models.

    Attributes:
        primary_series (np.ndarray): Input primary data series.
    """

    def __init__(self, primary_series: np.ndarray):
        """
        Args:
            primary_series (np.ndarray): Primary data series under investigation.
        """
        self.primary_series = primary_series

    @abstractmethod
    def train(self) -> None:
        """
        Abstract method to train the Bayesian model.

        This method should be overridden in all subclasses.
        """
        pass

    @abstractmethod
    def get_trace(self) -> pm.backends.base.MultiTrace:
        """
        Abstract method to retrieve the trace from the Bayesian model.

        This method should be overridden in all subclasses.

        Returns:
            pm.backends.base.MultiTrace: Trace of the model after training.
        """
        pass
