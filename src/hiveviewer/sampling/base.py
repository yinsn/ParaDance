from abc import ABCMeta, abstractmethod
from typing import List


class BaseSampler(metaclass=ABCMeta):
    """
    This class provides methods to sample data.
    """

    def __init__(self, sample_size: int) -> None:
        self.sample_size = sample_size

    @abstractmethod
    def sample(self, data: List[float]) -> None:
        raise NotImplementedError("Should implement sample()!")
