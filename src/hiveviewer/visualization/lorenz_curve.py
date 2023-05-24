from typing import List, Optional, Union

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
        self.unique_data = list(set(self.data))

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

    def get_bounds(
        self,
        num_quantiles: int,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        unique_bounds: bool = False,
    ) -> List[Union[int, float]]:
        """
        Get bounds from the data according to quantiles.

        :param num_quantiles: number of quantiles
        :param lower_bound: lower bound of the data
        :param upper_bound: upper bound of the data
        :param unique_bounds: whether to return unique bounds
        :return: a list of floats
        """
        bounds: List[Union[int, float]] = []
        data = self.slice_data(lower_bound, upper_bound)
        for i in range(1, num_quantiles):
            bounds.append(int(np.quantile(data, i / num_quantiles)))
        if unique_bounds:
            bounds = list(sorted(set(bounds)))
        return bounds

    @staticmethod
    def gini_coefficient(data: List[float]) -> float:
        """
        Compute Gini coefficient.

        :param data: a list of floats
        :return: Gini coefficient as a float.
        """
        n = len(data)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * data) - (n + 1) * np.sum(data)) / (n * np.sum(data))
        return float(gini)

    @staticmethod
    def plot_lorenz_curve(data: List[float], save_fig: bool = False) -> None:
        """
        Plot Lorenz curve.

        :param data: a list of floats
        :param save_fig: whether to save the figure
        :return: None
        """
        n = len(data)
        min_value = data[0]
        index = np.arange(1, n + 1) / n
        lorenz_curve = np.cumsum(data) / np.sum(data)
        plt.plot(index, lorenz_curve, color="orange", label="Lorenz Curve")
        plt.fill_between(index, lorenz_curve, index, color="orange", alpha=0.05)
        plt.xlabel(f"Cumulative Share of Population (truncated from {min_value})")
        plt.ylabel("Cumulative Share of Target Variable")
        gini = LorenzCurveGini.gini_coefficient(data)
        text_str = f"Gini: {gini:.4f}"
        plt.text(
            0.05,
            0.95,
            text_str,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
        )
        if save_fig:
            plt.savefig(f"gini_from_{min_value}.pdf", format="pdf")

    def lorenz_gini_from_to(
        self,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        save_fig: bool = False,
    ) -> float:
        """
        Plot Lorenz curve from lower_bound to upper_bound.

        :param lower_bound: lower bound of the data
        :param upper_bound: upper bound of the data
        :return: Gini coefficient as a float.
        """
        data = self.slice_data(lower_bound, upper_bound)
        self.plot_lorenz_curve(data=data, save_fig=save_fig)
        return self.gini_coefficient(data=data)

    def plot_lorenz_curves_with_lower_bounds(
        self,
        num_quantiles: int = 10,
        lower_bounds: Optional[List[float]] = None,
        slice_from: int = 0,
    ) -> List[float]:
        if lower_bounds is None:
            bounds = self.get_bounds(num_quantiles, lower_bound=slice_from)
        gini_list = []
        for bound in bounds:
            gini_list.append(self.lorenz_gini_from_to(lower_bound=bound))
        return gini_list

    def cal_gini_from_quantile(self, quantile: float) -> float:
        """
        Calculate Gini coefficient from quantile.

        :param quantile: quantile of the data
        :return: Gini coefficient as a float.
        """
        lower_bound = np.quantile(self.data, quantile)
        data = self.slice_data(lower_bound, upper_bound=None)
        return self.gini_coefficient(data=data)

    def gini_lower_bounds_curve(
        self,
        num_quantiles: int = 100,
        lower_bounds: Optional[List[float]] = None,
        slice_from: int = 0,
    ) -> List[float]:
        """
        Calculate and plot Gini coefficients with lower_bounds list.

        :param num_quantiles: number of quantiles
        :param lower_bounds: a list of lower bounds
        :param slice_from: slice data from slice_from
        :return: a list of Gini coefficients
        """
        gini_list = []
        if lower_bounds is None:
            lower_bounds = self.get_bounds(num_quantiles, lower_bound=slice_from)
        for lower_bound in lower_bounds:
            data = self.slice_data(lower_bound, upper_bound=None)
            gini_list.append(self.gini_coefficient(data=data))
        indice = np.arange(1, len(lower_bounds) + 1)

        _, ax1 = plt.subplots()
        ax1_color, ax2_color = "orange", "green"
        ax1.plot(indice, gini_list, color=ax1_color)
        ax2 = ax1.twinx()
        ax2.plot(indice, lower_bounds, color=ax2_color)
        ax1.tick_params(axis="y", colors=f"tab:{ax1_color}")
        ax2.tick_params(axis="y", colors=f"tab:{ax2_color}")
        ax1.set_xlabel("Quantiles")
        ax1.set_ylabel("Gini Coefficients", color=f"tab:{ax1_color}")
        ax2.set_ylabel("Lower Bounds", color=f"tab:{ax2_color}")
        plt.show()
        return gini_list
