import os
from typing import Dict, List, Optional, Union

import pandas as pd

from .base import BaseDataLoader


class CSVLoader(BaseDataLoader):
    "CSVLoader class for loading CSV files"

    def __init__(
        self,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        file_type: Optional[str] = "csv",
        max_rows: Optional[int] = None,
        clean_zero_columns: Optional[Union[bool, List]] = None,
        config: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            file_path, file_name, file_type, max_rows, clean_zero_columns, config
        )
        self.column_name_spliting()

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        if self.file_name is not None:
            file_url = os.path.join(str(self.file_path), self.file_name) + ".csv"
            df = pd.read_csv(file_url, low_memory=False)
        else:
            files = os.listdir(self.file_path)
            df_list = []
            for file in files:
                if file.endswith(str(self.file_type)):
                    file_url = os.path.join(str(self.file_path), file)
                    df_list.append(pd.read_csv(file_url, low_memory=False))
            df = pd.concat(df_list)
        if self.max_rows is not None:
            max_rows = min(self.max_rows, df.shape[0])
        else:
            max_rows = df.shape[0]
        return df.iloc[:max_rows, :]
