import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn3


class VennPloter:
    """
    Class to plot venn diagrams
    """

    def __init__(
        self, dataframe: pd.DataFrame, column1_name: str, column2_name: str
    ) -> None:
        """
        dataframe: dataframe to plot
        column1_name: name of the first column
        column2_name: name of the second column
        """
        self.df = dataframe
        self.column1_name = column1_name
        self.column2_name = column2_name
        self.column1_data = dataframe[column1_name]
        self.column2_data = dataframe[column2_name]
        self.df_len = len(dataframe)

    def get_conditions(self) -> tuple:
        """
        Returns the conditions for the venn diagram
        """
        both0 = (self.column1_data == 0) & (self.column2_data == 0)
        only_column1 = (self.column1_data == 1) & (self.column2_data == 0)
        only_column2 = (self.column1_data == 0) & (self.column2_data == 1)
        both1 = (self.column1_data == 1) & (self.column2_data == 1)
        return (both0, only_column1, only_column2, both1)

    def get_group_count_ratios(self) -> tuple:
        """
        Returns the ratios of the groups based on the count
        """
        both0, only_column1, only_column2, both1 = self.get_conditions()
        both0_ratio = sum(both0) / self.df_len
        only_column1_ratio = sum(only_column1) / self.df_len
        only_column2_ratio = sum(only_column2) / self.df_len
        both1_ratio = sum(both1) / self.df_len
        return (both0_ratio, only_column1_ratio, only_column2_ratio, both1_ratio)

    def get_group_value_ratios(self, value_column_name: str) -> tuple:
        """
        Returns the ratios of the groups based on the value column
        """
        both0, only_column1, only_column2, both1 = self.get_conditions()
        both0_sum = self.df[both0][value_column_name].sum()
        only_column1_sum = self.df[only_column1][value_column_name].sum()
        only_column2_sum = self.df[only_column2][value_column_name].sum()
        both1_sum = self.df[both1][value_column_name].sum()
        value_sum = self.df[value_column_name].sum()
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

    def plot_ratio_venn(self, ratios: tuple) -> None:
        """
        Plots the venn diagram with the ratios
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

        plt.show()

    def plot_count_ratio_venn(self) -> None:
        """
        Plots the venn diagram with the count ratios
        """
        ratios = self.get_group_count_ratios()
        self.plot_ratio_venn(ratios)

    def plot_value_ratio_venn(self, value_column_name: str) -> None:
        """
        Plots the venn diagram with the value ratios
        """
        ratios = self.get_group_value_ratios(value_column_name)
        self.plot_ratio_venn(ratios)
