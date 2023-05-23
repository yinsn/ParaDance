from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class LorenzCurve:
    def __init__(self, data: List[float]) -> None:
        """
        Initialize LorenzCurve object.

        Args:
            data: List or array containing the data points.

        """
        self.data = data

    def calculate_lorenz_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Lorenz curve coordinates.

        Returns:
            Two arrays representing the x and y coordinates of the Lorenz curve.

        """
        sorted_data = np.sort(self.data)
        cumulative_freq = np.cumsum(sorted_data) / np.sum(sorted_data)
        lorenz_curve_x = np.linspace(0, 1, len(cumulative_freq))
        lorenz_curve_y = np.cumsum(cumulative_freq)
        return lorenz_curve_x, lorenz_curve_y

    def calculate_gini_coefficient(self) -> float:
        """
        Calculate the Gini coefficient.

        Returns:
            The Gini coefficient value.

        """
        sorted_data = np.sort(self.data)
        cumulative_freq = np.cumsum(sorted_data) / np.sum(sorted_data)
        gini_coefficient = 1 - np.sum(
            (cumulative_freq[:-1] + cumulative_freq[1:])
            * (sorted_data[1:] - sorted_data[:-1])
        ) / (2 * np.sum(sorted_data))
        return float(gini_coefficient)

    def plot_lorenz_curve(self) -> None:
        """
        Plot the Lorenz curve.

        """
        lorenz_curve_x, lorenz_curve_y = self.calculate_lorenz_curve()

        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray"
        )  # 45-degree reference line
        plt.plot(lorenz_curve_x, lorenz_curve_y, label="Lorenz Curve")
        plt.fill_between(lorenz_curve_x, lorenz_curve_y, alpha=0.3)
        plt.xlabel("Cumulative Relative Frequency")
        plt.ylabel("Cumulative Share of Total")
        plt.title("Lorenz Curve")
        plt.legend()
        plt.show()

    def compute_and_return_gini_coefficient(self) -> float:
        """
        Compute and return the Gini coefficient.

        Returns:
            The Gini coefficient value.

        """
        return self.calculate_gini_coefficient()
