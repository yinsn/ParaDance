import os

import pandas as pd

from .base import BaseDataLoader


class ExcelLoader(BaseDataLoader):
    "ExcelLoader class for loading excel files"

    def __init__(self, file_path: str, **kwargs: str):
        super().__init__(file_path, **kwargs)
        self.column_name_spliting()

    def load_data(self) -> pd.DataFrame:
        """Load data from excel file."""
        if self.file_name is not None:
            file_url = os.path.join(self.file_path, self.file_name) + ".xlsx"
            return pd.read_excel(file_url)
        else:
            files = os.listdir(self.file_path)
            df_list = []
            for file in files:
                if file.endswith(self.file_type):
                    file_url = os.path.join(self.file_path, file)
                    df_list.append(pd.read_excel(file_url))
            return pd.concat(df_list)
