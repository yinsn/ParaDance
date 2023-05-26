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

    def clean_one_label_users(
        self,
        user_column: str = "user_id",
        label_column: str = "label",
        inline: bool = True,
    ) -> None:
        """Remove users with only one label."""
        if self.df is not None:
            valid_users = (
                self.df.groupby(user_column)
                .filter(lambda x: x[label_column].nunique() > 1)[user_column]
                .unique()
            )
            valid_df = self.df[self.df[user_column].isin(valid_users)].copy()
            if inline:
                self.df = valid_df
            else:
                self.valid_df = valid_df
