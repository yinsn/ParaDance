from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel


class BaseDataLoaderConfig(BaseModel):
    """Configuration model for data loaders.

    This class defines the configuration settings for data loaders, including file
    path, file name, file type, and maximum rows to be loaded. It utilizes Pydantic's
    BaseModel for data validation and settings management.

    Attributes:
        file_path (Optional[str]): The path to the file to be loaded. Defaults to None.
        file_name (Optional[str]): The name of the file to be loaded. Defaults to None.
        file_type (Optional[str]): The type of the file to be loaded. Defaults to 'csv'.
        max_rows (Optional[int]): The maximum number of rows to load from the file.
                                  Defaults to None, indicating no limit.
    """

    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = "csv"
    max_rows: Optional[int] = None
    clean_zero_columns: Optional[Union[bool, List]] = None


class BaseDataLoader(ABC):
    """Base class for data loaders."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        file_type: Optional[str] = "csv",
        max_rows: Optional[int] = None,
        clean_zero_columns: Optional[Union[bool, List]] = None,
        config: Optional[Dict] = None,
    ) -> None:
        if config is not None:
            self.config = BaseDataLoaderConfig(**config)
        else:
            self.config = BaseDataLoaderConfig(
                file_path=file_path,
                file_name=file_name,
                file_type=file_type,
                max_rows=max_rows,
                clean_zero_columns=clean_zero_columns,
            )

        self.file_path = self.config.file_path
        self.file_name = self.config.file_name
        self.file_type = self.config.file_type
        self.max_rows = self.config.max_rows
        self.clean_zero_columns = self.config.clean_zero_columns
        self.df = self.load_data()
        self.column_name_spliting()
        if self.clean_zero_columns is not None:
            self.clean_columns_zero(self.clean_zero_columns)

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data from file."""
        raise NotImplementedError("load_data() not implemented")

    def column_name_spliting(self, delimiter: str = ".") -> None:
        """Split column names by delimiter."""
        columns = []
        if self.df is not None:
            for column in self.df.columns:
                columns.append(column.split(delimiter)[-1])
            self.df.columns = pd.Index(columns)

    def add_one_smoothing(self, column: str) -> None:
        """Add one smoothing to a column.

        :param column: column name
        """
        if self.df is not None:
            self.df[column] = self.df[column] + 1

    def clean_columns_zero(self, columns: Union[bool, List] = False) -> None:
        """Clean columns with all zeros.

        :param columns: columns to clean
        """
        if self.df is not None and columns is not False:
            self.df[columns] = self.df[columns].fillna(0)
            self.df = self.df[(self.df[columns] > 0).all(axis=1)]
            self.df.reset_index(drop=True, inplace=True)
        self.df = self.df.fillna(0)

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
        df: pd.DataFrame,
        groupby: str,
        label_column: str,
        clip_column: Optional[str] = None,
    ) -> Tuple:
        """Clip and count with group.

        :param df: dataframe
        :param groupby: groupby column
        :param clip_column: column to clip
        :param label_column: label column
        """
        df_clean = BaseDataLoader.clean_one_label_users(
            df=df, user_column=groupby, label_column=label_column
        )
        if clip_column is not None:
            counts_df = BaseDataLoader.clip_and_sum_with_group(
                df=df_clean, groupby=groupby, clip_column=clip_column
            )
        else:
            counts_df = None
        return df_clean, counts_df
