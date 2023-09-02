from typing import Iterable, List, Union, cast

import matplotlib.pyplot as plt
import numpy as np

from ..evaluation.calculator import Calculator


class PortfolioPlotter:
    """PortfolioPlotter class for plotting portfolio curve."""

    def __init__(
        self,
        calculator: Calculator,
        target_column: str,
        points_num: int = 20,
        minimal_expected_return: float = 0.9,
        colors: List[str] = [
            "orange",
            "red",
            "green",
            "blue",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ],
    ) -> None:
        """Initialize PortfolioPlotter.

        Args:
            calculator (Calculator): Calculator instance for calculating the portfolio.
            target_column (str): target column for calculating the portfolio.
            points_num (int, optional): sample points number. Defaults to 20.
            minimal_expected_return (float, optional): minimal expected return ratio. Defaults to 0.9.
            colors (List[str], optional): colors for plotting.
        """
        self.calculator = calculator
        self.target_column = target_column
        self.points_num = points_num
        self.minimal_expected_return = minimal_expected_return
        self.colors = colors

    def _generate_points(self) -> None:
        """Generate points for plotting."""
        self.expected_returns = np.linspace(
            self.minimal_expected_return, 1, num=self.points_num, endpoint=True
        )
        self.top_ratios = [
            self.calculator.calculate_portfolio_concentration(
                target_column=self.target_column, expected_return=er
            )[1]
            for er in self.expected_returns
        ]
        self.expected_returns = self.expected_returns[::-1]
        self.top_ratios = self.top_ratios[::-1]
        self.top_ratios = [1 - ratio for ratio in self.top_ratios]
        self.expected_returns[0] = 1
        self.top_ratios[0] = 0

    def _plot_single(self, weights_for_equation: List[float], color: str) -> None:
        """Plot single portfolio curve.

        Args:
            weights_for_equation (List[float]): weights for equation
            color (str): color for plotting curve
        """
        self.calculator.get_overall_score(weights_for_equation)
        self._generate_points()
        plt.plot(self.top_ratios, self.expected_returns, color=color)
        plt.fill_between(
            self.top_ratios,
            self.expected_returns,
            y2=self.minimal_expected_return,
            color=color,
            alpha=0.1,
        )

    def plot(
        self, weights_for_equations: Union[List[float], List[List[float]]]
    ) -> None:
        """Plot portfolio curve.

        Args:
            weights_for_equations (Union[List[float], List[List[float]]]): weights for equations
        """
        plt.figure(figsize=(10, 6))
        if isinstance(weights_for_equations[0], Iterable):
            weights_for_equations = cast(List[List[float]], weights_for_equations)
            for i, weights_for_equation in enumerate(weights_for_equations):
                color = self.colors[i % len(self.colors)]
                self._plot_single(weights_for_equation, color=color)
        else:
            weights_for_equation = cast(List[float], weights_for_equations)
            self._plot_single(weights_for_equation, self.colors[0])
        plt.ylabel("Expected Return")
        plt.xlabel("Portfolio Efficiency")
        plt.title("Expected Return v.s. Portfolio Efficiency")
        plt.grid(True)
        plt.show()
