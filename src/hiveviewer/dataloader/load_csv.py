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

    def clip_and_sum_with_group(
        self, groupby: str, clip_column: str
    ) -> Optional[pd.Series]:
        """Clip and sum with group.

        :param groupby: groupby column
        :param clip_column: column to clip
        """
        if self.df is not None:
            threshold = self.df["live_gift_count"].quantile(0.999)
            self.df["clipped"] = self.df[clip_column].clip(upper=threshold)
            grouped = self.df.groupby(groupby)["clipped"].sum()
            return grouped
        else:
            return None

    def clean_one_label_users(
        self,
        user_column: str = "user_id",
        label_column: str = "label",
        inline: bool = True,
    ) -> None:
        """Remove users with only one label."""
        if self.df is not None:
            self.df = self.df.fillna(0)
            valid_users = (
                self.df.groupby(user_column)
                .filter(lambda x: x[label_column].nunique() > 1)[user_column]
                .unique()
            )
            valid_df = self.df[self.df[user_column].isin(valid_users)].copy()
            valid_df.reset_index(drop=True, inplace=True)
            if inline:
                self.df = valid_df
            else:
                self.valid_df = valid_df
