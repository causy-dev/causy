import datetime
import importlib
import json
from json import JSONEncoder
from typing import Dict, Any
import os

import torch
from pydantic import parse_obj_as

from causy.graph_utils import load_pipeline_steps_by_definition
from causy.interfaces import CausyAlgorithmReferenceType


def load_algorithm_from_reference(algorithm: str):
    st_function = importlib.import_module("causy.algorithms")
    st_function = getattr(st_function, algorithm)
    if not st_function:
        raise ValueError(f"Algorithm {algorithm} not found")
    return st_function


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
    from causy.interfaces import CausyAlgorithm

    return parse_obj_as(CausyAlgorithm, algorithm_dict)


def load_algorithm_by_reference(reference_type: str, algorithm: str):
    # TODO: test me
    if reference_type == CausyAlgorithmReferenceType.FILE:
        # validate if the reference points only in the same directory or subdirectory
        # to avoid security issues
        absolute_path = os.path.realpath(algorithm)
        common_prefix = os.path.commonprefix([os.getcwd(), absolute_path])
        if common_prefix != os.getcwd():
            raise ValueError("Invalid reference")

        with open(absolute_path, "r") as file:
            # load the algorithm from the file
            algorithm = json.loads(file.read())
        return load_algorithm_from_specification(algorithm)
    elif reference_type == CausyAlgorithmReferenceType.NAME:
        return load_algorithm_from_reference(algorithm)
    elif reference_type == CausyAlgorithmReferenceType.PYTHON_MODULE:
        st_function = importlib.import_module(algorithm)
        st_function = getattr(st_function, algorithm)
        if not st_function:
            raise ValueError(f"Algorithm {algorithm} not found")
        return st_function()


class CausyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if type(obj) is torch.Tensor:
            return None
        elif type(obj) is datetime.datetime:
            return obj.isoformat()
        return super().default(obj)
