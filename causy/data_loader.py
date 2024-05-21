import enum
import hashlib
import importlib
import json
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Union, Any, Optional

from pydantic import BaseModel

from causy.graph_utils import hash_dictionary


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
    options: Optional[Dict[str, Any]] = None


class AbstractDataLoader(ABC):
    @abstractmethod
    def __init__(self, reference: str, options: Optional[Dict[str, Any]] = None):
        pass

    reference: str
    options: Optional[Dict[str, Any]] = None

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

    def _hash_options(self):
        return hash_dictionary(self.options)


class FileDataLoader(AbstractDataLoader, ABC):
    """
    A data loader which loads data from a file reference (e.g. json, csv, etc.)
    """

    def __init__(self, reference: str, options: Optional[Dict[str, Any]] = None):
        self.reference = reference
        self.options = options

    reference: str

    def hash(self) -> str:
        with open(self.reference, "rb") as f:
            return (
                f'{hashlib.file_digest(f, "sha256").hexdigest()}_{self._hash_options()}'
            )


class JSONDataLoader(FileDataLoader):
    """
    A data loader which loads data from a json file
    """

    def load(self) -> Iterator[Dict[str, Union[float, int, str]]]:
        with open(self.reference, "r") as f:
            data = json.loads(f.read())
            if isinstance(data, list):
                for item in data:
                    yield item
            elif isinstance(data, dict):
                yield {"_dict": data}
                return
            else:
                raise ValueError(
                    f"Invalid JSON format. Data in {self.reference} is of type {type(data)}."
                )


class JSONLDataLoader(FileDataLoader):
    """
    A data loader which loads data from a jsonl file
    """

    def load(self) -> Iterator[Dict[str, Union[float, int, str]]]:
        with open(self.reference, "r") as f:
            for line in f:
                yield json.loads(line)


class DynamicDataLoader(AbstractDataLoader):
    """
    A data loader which loads another data loader dynamically based on the reference
    """

    def __init__(self, reference: str, options: Optional[Dict[str, Any]] = None):
        self.reference = reference
        self.data_loader = self._load_data_loader()
        self.options = options

    reference: str
    data_loader: AbstractDataLoader

    def _load_data_loader(self) -> AbstractDataLoader:
        module = importlib.import_module(self.reference)
        # todo: should the cls be referenced here?
        return module.DataLoader(**self.options)

    def load(self) -> Iterator[Dict[str, Union[float, int, str]]]:
        return self.data_loader.load()


DATA_LOADERS = {
    DataLoaderType.JSON: JSONDataLoader,
    DataLoaderType.JSONL: JSONLDataLoader,
    DataLoaderType.DYNAMIC: DynamicDataLoader,
}


def load_data_loader(reference: DataLoaderReference) -> AbstractDataLoader:
    """
    loads the data loader based on the reference
    :param reference: a data loader reference
    :return:
    """

    return DATA_LOADERS[reference.type](reference.reference, reference.options)
