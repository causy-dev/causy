import enum
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Union

from pydantic import BaseModel


class DataLoaderType(enum.StrEnum):
    DYNAMIC = "dynamic"  # python function which yields data
    JSON = "json"
    JSONL = "jsonl"


class DataLoaderReference(BaseModel):
    """represents a single data loader
    :param type: the type of dataloader
    :param reference: path to either the python class which can be executed to load the data or the data source file itself
    """

    type: DataLoaderType
    reference: str


class AbstractDataLoader(ABC):
    @abstractmethod
    def load(self) -> Iterator[Dict[str, Union[float, int, str]]]:
        """
        loads the data from the source and returns it as an iterator
        :return:
        """
        pass

    @abstractmethod
    def hash(self) -> str:
        """
        returns a hash of the data that is loaded
        :return:
        """
        pass
