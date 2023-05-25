from abc import ABCMeta, abstractmethod
from typing import List, Optional


class BaseSampler(metaclass=ABCMeta):
    """
    This class provides methods to sample data.
    """

    def __init__(
        self,
        sample_size: int,
        data: List[float],
        slice_from: Optional[float] = None,
        slice_to: Optional[float] = None,
    ) -> None:
        self.sample_size = sample_size
        self.data = data
        self.slice_from = slice_from
        self.slice_to = slice_to

    @abstractmethod
    def sample(self) -> List[float]:
        raise NotImplementedError("Should implement sample()!")
