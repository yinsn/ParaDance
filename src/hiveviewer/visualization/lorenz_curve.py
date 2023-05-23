from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


class LorenzCurveGini:
    """
    This class provides methods to compute Gini coefficient and plot Lorenz curve.
    """

    def __init__(self, data: List[float]):
        """
        Initialize with a list of data.

        :param data: a list of floats
        """
        self.data = sorted(data)

    def slice_data(
        self, lower_bound: Optional[float], upper_bound: Optional[float]
    ) -> List[float]:
        """
        Slice data from lower_bound to upper_bound.

        :param lower_bound: lower bound of the data
        :param upper_bound: upper bound of the data
        :return: a list of floats
        """
        if lower_bound is None and upper_bound is None:
            data = self.data
        elif lower_bound is None and upper_bound is not None:
            data = [x for x in self.data if x <= upper_bound]
        elif upper_bound is None and lower_bound is not None:
            data = [x for x in self.data if lower_bound <= x]
        elif lower_bound is not None and upper_bound is not None:
            data = [x for x in self.data if lower_bound <= x <= upper_bound]
        return data

    def gini_coefficient(
        self, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None
    ) -> float:
        """
        Compute Gini coefficient.

        :param lower_bound: lower bound of the data
        :param upper_bound: upper bound of the data
        :return: Gini coefficient as a float.
        """
        data = self.slice_data(lower_bound, upper_bound)
        n = len(data)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * data) - (n + 1) * np.sum(data)) / (n * np.sum(data))
        return float(gini)

    @staticmethod
    def plot_lorenz_curve(data: List[float]) -> None:
        """
        Plot Lorenz curve.

        :return: None
        """
        n = len(data)
        index = np.arange(1, n + 1) / n
        lorenz_curve = np.cumsum(data) / np.sum(data)
        plt.plot(index, lorenz_curve, color="orange", label="Lorenz Curve")
        plt.fill_between(index, lorenz_curve, index, color="orange", alpha=0.05)
        plt.xlabel("Cumulative Share of Population")
        plt.ylabel("Cumulative Share of Target Variable")
        plt.show()

    def lorenz_gini_from_to(
        self, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None
    ) -> float:
        """
        Plot Lorenz curve from lower_bound to upper_bound.

        :param lower_bound: lower bound of the data
        :param upper_bound: upper bound of the data
        :return: Gini coefficient as a float.
        """
        data = self.slice_data(lower_bound, upper_bound)
        self.plot_lorenz_curve(data)
        return self.gini_coefficient(lower_bound, upper_bound)
