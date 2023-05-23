from typing import List

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

    def gini_coefficient(self) -> float:
        """
        Compute Gini coefficient.

        :return: Gini coefficient as a float.
        """
        n = len(self.data)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * self.data) - (n + 1) * np.sum(self.data)) / (
            n * np.sum(self.data)
        )
        return float(gini)

    def plot_lorenz_curve(self) -> None:
        """
        Plot Lorenz curve.

        :return: None
        """
        n = len(self.data)
        index = np.arange(1, n + 1) / n
        lorenz_curve = np.cumsum(self.data) / np.sum(self.data)
        plt.plot(index, lorenz_curve, color="orange", label="Lorenz Curve")
        plt.fill_between(index, lorenz_curve, index, color="orange", alpha=0.05)
        plt.title("Lorenz Curve")
        plt.xlabel("Cumulative Share of Population")
        plt.ylabel("Cumulative Share of Target Variable")
        plt.show()
