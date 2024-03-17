import enum
from typing import Optional, Dict, List, Any

from pydantic import BaseModel


class ComparisonSettings(BaseModel):
    """
    Represents comparison settings
    :param min: minimum amount of nodes to generate combinations from
    :param max: maximum amount of nodes to generate combinations from
    """

    min: int = 2
    max: int = 0  # stands for as many as fields


class Generator(BaseModel):
    """
    Represents a generator
     :param reference: a reference to the generator python path
     :param comparison_settings: settings of the generator
     :param options: generator specific options
    """

    reference: str  # reference to the python path
    comparison_settings: ComparisonSettings
    options: Dict[str, Any]


class PipelineStep(BaseModel):
    """
    Represents a single pipeline step
    :param reference: a reference to the pipeline python path
    :param number_of_comparison_elements: number of elements that should be compared to eachother (0 = as many as possible)
    :param generator: the generator which should be used
    """

    reference: str  # reference to the python path
    number_of_comparison_elements: int = 0
    generator: Generator


class Pipeline(BaseModel):
    """
    Represents a pipeline
    :param name: custom name for the pipeline
    :param reference: is a reference to an existing pipeline in causy core or an external pipeline definition (in python/yaml)
    :param steps: an inline pipeline definition
    """

    name: str
    reference: Optional[str] = None
    steps: Optional[List[PipelineStep]] = None


class DataLoaderType(enum.StrEnum):
    DYNAMIC = "dynamic"  # python function which yields data
    JSON = "json"
    JSONL = "jsonl"


class DataLoader(BaseModel):
    """represents a single data loader
    :param type: the type of dataloader
    :param path: path to either the python class which can be executed to load the data or the data source file itself
    """

    type: DataLoaderType
    path: str


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

    pipelines: Dict[str, Pipeline]
    data_loaders: Dict[str, Pipeline]
    experiments: Dict[str, Experiment]
