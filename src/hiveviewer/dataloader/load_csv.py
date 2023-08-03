import os

import pandas as pd

from .base import BaseDataLoader


class CSVLoader(BaseDataLoader):
    "CSVLoader class for loading CSV files"

    def __init__(self, file_path: str, **kwargs: str):
        super().__init__(file_path, **kwargs)
        self.column_name_spliting()

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        if self.file_name is not None:
            file_url = os.path.join(self.file_path, self.file_name) + ".csv"
            return pd.read_csv(file_url, low_memory=False)
        else:
            files = os.listdir(self.file_path)
            df_list = []
            for file in files:
                if file.endswith(self.file_type):
                    file_url = os.path.join(self.file_path, file)
                    df_list.append(pd.read_csv(file_url, low_memory=False))
            return pd.concat(df_list)
