import copy
import datetime
import importlib
import json
from json import JSONEncoder
from typing import Dict, Any, List
import os

import torch
import yaml
from pydantic import TypeAdapter


from causy.edge_types import EDGE_TYPES
from causy.graph_utils import load_pipeline_steps_by_definition
from causy.models import AlgorithmReferenceType, Result, AlgorithmReference
from causy.variables import deserialize_variable
from causy.causal_discovery import AVAILABLE_ALGORITHMS


def serialize_algorithm(model, algorithm_name: str = None):
    """Serialize the model into a dictionary."""

    if algorithm_name:
        model.algorithm.name = algorithm_name
    return model.algorithm.model_dump()


def load_algorithm_from_specification(algorithm_dict: Dict[str, Any]):
    """Load the model from a dictionary."""
    algorithm_dict["pipeline_steps"] = load_pipeline_steps_by_definition(
        algorithm_dict["pipeline_steps"]
    )
    if "extensions" in algorithm_dict and algorithm_dict["extensions"] is not None:
        algorithm_dict["extensions"] = load_pipeline_steps_by_definition(
            algorithm_dict["extensions"]
        )
    else:
        algorithm_dict["extensions"] = []
    if "variables" not in algorithm_dict or algorithm_dict["variables"] is None:
        algorithm_dict["variables"] = []

    algorithm_dict["variables"] = [
        deserialize_variable(variable) for variable in algorithm_dict["variables"]
    ]
    from causy.models import Algorithm

    return TypeAdapter(Algorithm).validate_python(algorithm_dict)


def load_algorithm_by_reference(reference_type: str, algorithm: str):
    # TODO: test me
    if reference_type == AlgorithmReferenceType.FILE:
        # validate if the reference points only in the same directory or subdirectory
        # to avoid security issues
        absolute_path = os.path.realpath(algorithm)
        common_prefix = os.path.commonprefix([os.getcwd(), absolute_path])
        if common_prefix != os.getcwd():
            raise ValueError("Invalid reference")

        with open(absolute_path, "r") as file:
            # load the algorithm from the file
            # try first json
            try:
                algorithm = json.loads(file.read())
                return load_algorithm_from_specification(algorithm)
            except json.JSONDecodeError:
                pass
            file.seek(0)
            # then try yaml
            try:
                data = yaml.load(file.read(), Loader=yaml.FullLoader)
                return load_algorithm_from_specification(data)
            except yaml.YAMLError:
                pass
            raise ValueError("Invalid file format")

    elif reference_type == AlgorithmReferenceType.NAME:
        return copy.deepcopy(AVAILABLE_ALGORITHMS[algorithm]()._original_algorithm)
    elif reference_type == AlgorithmReferenceType.PYTHON_MODULE:
        module_, ref_ = algorithm.rsplit(".", 1)
        module = importlib.import_module(module_)
        st_function = getattr(module, ref_)
        if not st_function:
            raise ValueError(f"Algorithm {algorithm} not found")
        return st_function()._original_algorithm


class CausyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if type(obj) is torch.Tensor:
            return None
        elif type(obj) is datetime.datetime:
            return obj.isoformat()
        return super().default(obj)


def load_json(pipeline_file: str):
    with open(pipeline_file, "r") as file:
        pipeline = json.loads(file.read())
    return pipeline


def deserialize_result(result: Dict[str, Any], klass=Result):
    """Deserialize the result."""

    result["algorithm"] = AlgorithmReference(**result["algorithm"])
    for i, edge in enumerate(result["edges"]):
        result["edges"][i]["edge_type"] = EDGE_TYPES[edge["edge_type"]["name"]](
            **edge["edge_type"]
        )

    return TypeAdapter(klass).validate_python(result)
