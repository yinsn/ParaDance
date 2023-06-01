import os
from typing import Optional

import pandas as pd

from .base import BaseDataLoader


class CSVLoader(BaseDataLoader):
    "CSVLoader class for loading CSV files"

    def __init__(self, file_path: str, **kwargs: str):
        super().__init__(file_path, **kwargs)
        self.column_name_spliting()

    def load_data(self) -> Optional[pd.DataFrame]:
        """Load data from CSV file."""
        file_url = os.path.join(self.file_path, self.file_name) + ".csv"
        return pd.read_csv(file_url)

    def column_name_spliting(self, delimiter: str = ".") -> None:
        """Split column names by delimiter."""
        columns = []
        if self.df is not None:
            for column in self.df.columns:
                columns.append(column.split(delimiter)[-1])
            self.df.columns = columns

    def add_one_smoothing(self, column: str) -> None:
        """Add one smoothing to a column.

        :param column: column name
        """
        if self.df is not None:
            self.df[column] = self.df[column] + 1

    @staticmethod
    def clip_and_sum_with_group(
        df: pd.DataFrame, groupby: str, clip_column: str
    ) -> Optional[pd.Series]:
        """Clip and sum with group.

        :param df: dataframe
        :param groupby: groupby column
        :param clip_column: column to clip
        """
        if df is not None:
            threshold = df[clip_column].quantile(0.999)
            df["clipped"] = df[clip_column].clip(upper=threshold)
            grouped = df.groupby(groupby)["clipped"].sum()
            return grouped + 1  # add one smoothing
        else:
            return None

    @staticmethod
    def clean_one_label_users(
        df: pd.DataFrame,
        user_column: str = "user_id",
        label_column: str = "label",
    ) -> pd.DataFrame:
        """Remove users with only one label.

        :param df: dataframe
        :param user_column: user column name
        :param label_column: label column name
        """
        df = df.fillna(0)
        valid_users = (
            df.groupby(user_column)
            .filter(lambda x: x[label_column].nunique() > 1)[user_column]
            .unique()
        )
        valid_df = df[df[user_column].isin(valid_users)].copy()
        valid_df.reset_index(drop=True, inplace=True)
        return valid_df

    @staticmethod
    def clip_clean_count_with_group(
        df: pd.DataFrame, groupby: str, clip_column: str, label_column: str
    ) -> tuple:
        """Clip and count with group.

        :param df: dataframe
        :param groupby: groupby column
        :param clip_column: column to clip
        :param label_column: label column
        """
        df_clean = CSVLoader.clean_one_label_users(
            df=df, user_column=groupby, label_column=label_column
        )
        counts_df = CSVLoader.clip_and_sum_with_group(
            df=df_clean, groupby=groupby, clip_column=clip_column
        )
        return df_clean, counts_df
