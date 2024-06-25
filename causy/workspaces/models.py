from typing import Optional, Dict, Any

from pydantic import BaseModel

from causy.data_loader import DataLoaderReference
from causy.models import AlgorithmReference, Algorithm


class Experiment(BaseModel):
    """
    represents a single experiment
    :param name: name of the experiment
    :param pipeline: the name of the pipeline used
    """

    pipeline: str
    dataloader: str
    variables: Optional[Dict[str, Any]] = None


class Workspace(BaseModel):
    name: str
    author: Optional[str]

    pipelines: Optional[Dict[str, Algorithm | AlgorithmReference]]
    dataloaders: Optional[Dict[str, DataLoaderReference]]
    experiments: Optional[Dict[str, Experiment]]
