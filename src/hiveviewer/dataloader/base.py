from abc import ABC, abstractmethod

import pandas as pd


class BaseDataLoader(ABC):
    """Base class for data loaders."""

    def __init__(self, file_path: str, file_name: str, file_type: str = "csv") -> None:
        self.file_path = file_path
        self.file_type = file_type
        self.file_name = file_name
        self.df = self.load_data()

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data from file."""
        raise NotImplementedError("load_data() not implemented")
