from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pandas as pd


class BaseVennPloter(ABC):
    """Base class for venn ploter."""

    def __init__(
        self, dataframe: pd.DataFrame, column1_name: str, column2_name: str
    ) -> None:
        self.df = dataframe
        self.column1_name = column1_name
        self.column2_name = column2_name
        self.column1_data = dataframe[column1_name]
        self.column2_data = dataframe[column2_name]
        self.df_len = len(dataframe)

    @abstractmethod
    def get_conditions(self) -> Tuple:
        """Returns the conditions for the venn diagram"""
        raise NotImplementedError("get_conditions() not implemented")

    @abstractmethod
    def get_group_count_ratios(self) -> Tuple:
        """Returns the ratios of the groups based on the count"""
        raise NotImplementedError("get_group_count_ratios() not implemented")

    @abstractmethod
    def get_group_value_ratios(
        self,
        value_column_name: str,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> Tuple:
        """Returns the ratios of the groups based on the value

        value_column_name: name of the column with the values
        lower_bound: lower bound of the values
        upper_bound: upper bound of the values
        """
        raise NotImplementedError("get_group_value_ratios() not implemented")

    @abstractmethod
    def plot_ratio_venn(
        self,
        ratios: tuple,
        save_fig: bool = False,
        file_tag: Optional[float] = None,
        file_type: str = "pdf",
    ) -> None:
        """Plots the venn diagram based on the ratios"""
        raise NotImplementedError("plot_ratio_venn() not implemented")

    def plot_count_ratio_venn(self) -> None:
        """
        Plots the venn diagram with the count ratios
        """
        ratios = self.get_group_count_ratios()
        self.plot_ratio_venn(ratios)

    def plot_value_ratio_venn(
        self,
        value_column_name: str,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        save_fig: bool = False,
        file_type: str = "pdf",
    ) -> None:
        """
        Plots the venn diagram with the value ratios

        value_column_name: name of the column with the values
        lower_bound: lower bound of the values
        upper_bound: upper bound of the values
        save_fig: whether to save the figure
        """

        ratios = self.get_group_value_ratios(
            value_column_name, lower_bound=lower_bound, upper_bound=upper_bound
        )
        self.plot_ratio_venn(
            ratios, save_fig=save_fig, file_tag=upper_bound, file_type=file_type
        )
