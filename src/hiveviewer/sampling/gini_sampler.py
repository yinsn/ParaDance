from typing import List, Optional

from ..visualization.lorenz_curve import LorenzCurveGini
from .base import BaseSampler


class GiniSampler(BaseSampler):
    """
    This class provides methods to sample data according to Gini coefficient.
    """

    def __init__(
        self,
        sample_size: int,
        data: List[float],
        slice_from: Optional[float] = None,
        slice_to: Optional[float] = None,
    ) -> None:
        super().__init__(sample_size, data, slice_from, slice_to)
        self.lorenz_gini = LorenzCurveGini(self.data)
        self.bounds = self.lorenz_gini.get_bounds(
            num_quantiles=self.sample_size,
            slice_from=self.slice_from,
            slice_to=self.slice_to,
            unique_bounds=True,
        )

    def sample(self) -> List[float]:
        """
        Sample data according to Gini coefficient.
        """
        gini_list = self.lorenz_gini.get_gini_list_from_bounds(self.bounds)
        return gini_list
