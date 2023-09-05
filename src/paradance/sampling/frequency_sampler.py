from collections import Counter
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .base import BaseSampler


class FrequencySampler(BaseSampler):
    """
    This class provides method to sample data by frequency.
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
        super().__init__(
            sample_size, data, slice_from, slice_to, log_scale, laplace_smoothing
        )

    def sample(self) -> dict:
        if self.slice_from is not None:
            self.data = [x for x in self.data if x > self.slice_from]
        if self.slice_to is not None:
            self.data = [x for x in self.data if x <= self.slice_to]

        # calculate the percentiles that will give us the required sample size
        percentiles = np.linspace(0, 100, self.sample_size + 2)[1:-1]
        samples = np.percentile(self.data, percentiles)
        if self.log_scale:
            samples = [np.exp(x) for x in samples]
        else:
            samples = [x for x in samples]
        if self.laplace_smoothing:
            samples = [x - 1 for x in samples]
        return dict(Counter(samples))
