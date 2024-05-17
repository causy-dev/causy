import enum
from typing import Optional, Dict, List, Any

from pydantic import BaseModel

from causy.interfaces import CausyAlgorithm, CausyAlgorithmReference


class DataLoaderType(enum.StrEnum):
    DYNAMIC = "dynamic"  # python function which yields data
    JSON = "json"
    JSONL = "jsonl"


class DataLoader(BaseModel):
    """represents a single data loader
    :param type: the type of dataloader
    :param reference: path to either the python class which can be executed to load the data or the data source file itself
    """

    type: DataLoaderType
    reference: str


class Experiment(BaseModel):
    """
    represents a single experiment
    :param name: name of the experiment
    :param pipeline: the name of the pipeline used
    """

    pipeline: str
    data_loader: str


class Workspace(BaseModel):
    name: str
    author: Optional[str]

    pipelines: Optional[Dict[str, CausyAlgorithm | CausyAlgorithmReference]]
    data_loaders: Optional[Dict[str, DataLoader]]
    experiments: Optional[Dict[str, Experiment]]
