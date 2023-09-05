from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class BaseSampler(metaclass=ABCMeta):
    """
    This class provides methods to sample data.
    """

    def __init__(
        self,
        sample_size: int,
        data: Union[pd.Series, List[float]],
        slice_from: Optional[float] = None,
        slice_to: Optional[float] = None,
        log_scale: Optional[bool] = True,
        laplace_smoothing: Optional[bool] = True,
    ) -> None:
        self.sample_size = sample_size
        self.data = data
        self.slice_from = slice_from
        self.slice_to = slice_to
        self.log_scale = log_scale
        self.laplace_smoothing = laplace_smoothing
        if self.laplace_smoothing:
            self.data = [x + 1 for x in self.data]
        if self.log_scale:
            self.data = [np.log(x) for x in self.data]
        self.boundary_dict = self.sample()

    @abstractmethod
    def sample(self) -> dict:
        raise NotImplementedError("Should implement sample()!")
