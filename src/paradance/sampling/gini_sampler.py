from collections import Counter
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ..visualization.lorenz_curve import LorenzCurveGini
from .base import BaseSampler


class GiniSampler(BaseSampler):
    """
    This class provides methods to sample data according to Gini coefficient.
    """

    def __init__(
        self,
        sample_size: int,
        data: Union[pd.Series, List[float]],
        slice_from: Optional[float] = None,
        slice_to: Optional[float] = None,
        log_scale: Optional[bool] = True,
        laplace_smoothing: Optional[bool] = True,
        bounds_num: Optional[int] = None,
    ) -> None:
        """
        Initialize with a list of data.

        :param sample_size: number of samples
        :param data: a list of floats
        :param slice_from: lower bound of the data
        :param slice_to: upper bound of the data
        :param bounds_num: number of bounds
        :return: a list of floats
        """
        super().__init__(
            sample_size, data, slice_from, slice_to, log_scale, laplace_smoothing
        )
        self.lorenz_gini = LorenzCurveGini(self.data)
        if bounds_num is None:
            self.bounds_num = self.sample_size * 10
        else:
            self.bounds_num = bounds_num
        self.bounds = self.lorenz_gini.get_bounds(
            num_quantiles=self.bounds_num,
            slice_from=self.slice_from,
            slice_to=self.slice_to,
            unique_bounds=False,
        )

    def equidistant_indices(self, gini_list: List[float]) -> List[float]:
        """
        Get equidistant indices from Gini list.

        :param gini_list: a list of Gini coefficients
        """
        min_gini = np.min(gini_list)
        max_gini = np.max(gini_list)

        gini_segments = np.linspace(max_gini, min_gini, self.sample_size + 1)
        start_idx = 0

        segment_indices = []
        for g in gini_segments:
            segment_idx = start_idx + np.abs(gini_list[start_idx:] - g).argmin()
            segment_indices.append(segment_idx)
            start_idx = segment_idx

        segment_bounds = [self.bounds[i] for i in segment_indices]

        return segment_bounds

    def sample(self) -> dict:
        """
        Sample data according to Gini coefficient.
        """

        gini_list = self.lorenz_gini.get_gini_list_from_bounds(self.bounds)
        segment_bounds = self.equidistant_indices(gini_list)
        return dict(Counter(segment_bounds))
