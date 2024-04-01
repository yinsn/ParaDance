from typing import List

from mixician import SelfBalancingLogarithmPCACalculator

from .base_calculator import BaseCalculator


class LogarithmPCACalculator(BaseCalculator):
    """A calculator for performing PCA (Principal Component Analysis) operations.

    This calculator uses an instance of a SelfBalancingLogarithmPCACalculator to perform
    the underlying PCA calculations and updates.

    Attributes:
        pca_calculator (SelfBalancingLogarithmPCACalculator): The PCA calculator instance
            used for performing PCA operations.
        df (DataFrame): A copy of the cleaned dataframe from the `pca_calculator`.
    """

    def __init__(self, pca_calculator: SelfBalancingLogarithmPCACalculator):
        super().__init__(selected_columns=pca_calculator.selected_columns)
        self.pca_calculator = pca_calculator
        self.df = self.pca_calculator.clean_dataframe.copy()
        self.df_len = len(self.df)
        self.equation_type = "log_pca"

    def get_overall_score(
        self,
        weights_for_equation: List[float],
    ) -> None:
        """
        Calculates and assigns an overall score to each entry in the dataframe based on
        the provided weights for the PCA equation.

        This method updates the internal dataframe `df` with a new column `overall_score`
        that contains the calculated scores for each entry.

        Args:
            weights_for_equation (List[float]): A list of weights for calculating the
                overall PCA score.
        """
        self.pca_calculator.update(
            pca_weights=weights_for_equation,
        )
        self.df["overall_score"] = self.pca_calculator.cumulative_product_scores
