import os
from typing import List, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn3

from .venn_base import BaseVennPloter


class Venn3Ploter(BaseVennPloter):
    """
    Class to plot a venn diagram with 4 groups
    """

    def __init__(
        self, dataframe: pd.DataFrame, column1_name: str, column2_name: str
    ) -> None:
        super().__init__(dataframe, column1_name, column2_name)

    def get_conditions(self) -> Tuple:
        """
        Returns the conditions for the venn diagram
        """
        both0 = (self.column1_data == 0) & (self.column2_data == 0)
        only_column1 = (self.column1_data == 1) & (self.column2_data == 0)
        only_column2 = (self.column1_data == 0) & (self.column2_data == 1)
        both1 = (self.column1_data == 1) & (self.column2_data == 1)
        return (both0, only_column1, only_column2, both1)

    def get_group_count_ratios(self) -> Tuple:
        """
        Returns the ratios of the groups based on the count
        """
        both0, only_column1, only_column2, both1 = self.get_conditions()
        both0_ratio = sum(both0) / self.df_len
        only_column1_ratio = sum(only_column1) / self.df_len
        only_column2_ratio = sum(only_column2) / self.df_len
        both1_ratio = sum(both1) / self.df_len
        return (both0_ratio, only_column1_ratio, only_column2_ratio, both1_ratio)

    def get_group_value_ratios(
        self,
        value_column_name: str,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> Tuple:
        """
        Returns the ratios of the groups based on the value

        value_column_name: name of the column with the values
        lower_bound: lower bound of the values
        upper_bound: upper bound of the values
        """
        if lower_bound is None:
            greater_than_lower_bound = True
        else:
            greater_than_lower_bound = self.df[value_column_name] > lower_bound
        if upper_bound is None:
            lower_than_upper_bound = True
        else:
            lower_than_upper_bound = self.df[value_column_name] < upper_bound

        both0, only_column1, only_column2, both1 = self.get_conditions()
        both0 = both0 & greater_than_lower_bound & lower_than_upper_bound
        only_column1 = only_column1 & greater_than_lower_bound & lower_than_upper_bound
        only_column2 = only_column2 & greater_than_lower_bound & lower_than_upper_bound
        both1 = both1 & greater_than_lower_bound & lower_than_upper_bound
        both0_sum = self.df[both0][value_column_name].sum()
        only_column1_sum = self.df[only_column1][value_column_name].sum()
        only_column2_sum = self.df[only_column2][value_column_name].sum()
        both1_sum = self.df[both1][value_column_name].sum()
        value_sum = both0_sum + only_column1_sum + only_column2_sum + both1_sum
        both0_sum_ratio = both0_sum / value_sum
        only_column1_sum_ratio = only_column1_sum / value_sum
        only_column2_sum_ratio = only_column2_sum / value_sum
        both1_sum_ratio = both1_sum / value_sum
        return (
            float(both0_sum_ratio),
            float(only_column1_sum_ratio),
            float(only_column2_sum_ratio),
            float(both1_sum_ratio),
        )

    def plot_ratio_venn(
        self,
        ratios: tuple,
        save_fig: bool = False,
        file_tag: Optional[float] = None,
        file_type: str = "pdf",
    ) -> None:
        """
        Plots the venn diagram with the ratios

        ratios: ratios of the groups
        save_fig: whether to save the figure
        file_tag: tag to add to the file name
        file_type: file type to save the figure
        """
        both0_ratio, only_column1_ratio, only_column2_ratio, both1_ratio = ratios
        v = venn3(
            subsets=(
                0,
                0,
                0,
                both0_ratio,
                only_column1_ratio,
                only_column2_ratio,
                both1_ratio,
            ),
            set_labels=(self.column1_name, self.column2_name, "all"),
        )
        v.get_label_by_id("101").set_text(
            f"{only_column1_ratio:.3%}\n{self.column1_name} && !{self.column2_name}"
        )
        v.get_label_by_id("011").set_text(
            f"{only_column2_ratio:.3%}\n!{self.column1_name} && {self.column2_name}"
        )
        v.get_label_by_id("001").set_text(
            f"\n\n\n\n\n\n{both0_ratio:.3%}\n!{self.column1_name} && !{self.column2_name}"
        )
        v.get_label_by_id("111").set_text(
            f"\n\n\n\n\n\n\n\n{both1_ratio:.3%}\n{self.column1_name} && {self.column2_name}"
        )

        v.get_patch_by_id("101").set_color("orange")
        v.get_patch_by_id("011").set_color("green")
        v.get_patch_by_id("111").set_color("yellowgreen")
        v.get_patch_by_id("001").set_color("skyblue")

        if save_fig:
            if file_tag is None:
                file_name = f"ratio_venn.{file_type}"
            else:
                text_str: str = f"upper bound: {file_tag}"
                plt.text(
                    0.90,
                    0.05,
                    text_str,
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment="top",
                )
                file_name = f"ratio_venn_{file_tag}.{file_type}"
            plt.savefig(file_name, format=file_type)

    def plot_value_ratio_venn_with_upper_bounds(
        self,
        value_column_name: str,
        upper_bounds: List[float],
        duration: Optional[int] = 300,
    ) -> None:
        filenames: List[str] = []
        for upper_bound in upper_bounds:
            filename = f"ratio_venn_{upper_bound}.png"
            filenames.append(filename)

        for upper_bound in upper_bounds:
            self.plot_value_ratio_venn(
                value_column_name,
                lower_bound=None,
                upper_bound=upper_bound,
                save_fig=True,
                file_type="png",
            )
            plt.close()
        with imageio.get_writer(
            f"ratio_venn_bounds_from_{upper_bounds[0]}_to_{upper_bounds[-1]}.gif",
            mode="I",
            duration=duration,
            loop=0,
        ) as writer:
            for filename in filenames:
                image = imageio.v2.imread(filename, pilmode="RGBA")
                writer.append_data(image)
        for filename in set(filenames):
            os.remove(filename)
