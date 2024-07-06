from typing import Optional, Dict

from pydantic import BaseModel

from causy.data_loader import DataLoaderReference
from causy.models import AlgorithmReference, Algorithm
from causy.variables import VariableType


class Experiment(BaseModel):
    """
    represents a single experiment
    :param name: name of the experiment
    :param pipeline: the name of the pipeline used
    """

    pipeline: str
    dataloader: str
    variables: Optional[Dict[str, VariableType]] = {}


class Workspace(BaseModel):
    name: str
    author: Optional[str]

    pipelines: Optional[Dict[str, Algorithm | AlgorithmReference]]
    dataloaders: Optional[Dict[str, DataLoaderReference]]
    experiments: Optional[Dict[str, Experiment]]
