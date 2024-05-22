import hashlib
import importlib
import json
from typing import List, Tuple, Dict

from causy.variables import deserialize_variable_references


def unpack_run(args):
    tst = args[0]
    del args[0]
    return tst(*args)


def serialize_module_name(cls):
    return f"{cls.__class__.__module__}.{cls.__class__.__name__}"


def load_pipeline_artefact_by_definition(step):
    name = step["name"]
    path = ".".join(name.split(".")[:-1])
    cls = name.split(".")[-1]
    st_function = importlib.import_module(path)
    st_function = getattr(st_function, cls)
    if not st_function:
        raise ValueError(f"{name} not found")

    del step["name"]

    return st_function(**step)


def load_pipeline_steps_by_definition(steps):
    pipeline = []
    for step in steps:
        st_function = load_pipeline_artefact_by_definition(step)
        st_function = deserialize_variable_references(st_function)
        pipeline.append(st_function)
    return pipeline


def retrieve_edges(graph) -> List[Tuple[str, str]]:
    """
    Returns a list of edges from the graph
    :param graph: a graph
    :return: a list of edges
    """
    edges = []
    for u in graph.edges:
        for v in graph.edges[u]:
            edges.append((u, v))
    return edges


def hash_dictionary(dct: Dict):
    """
    Hash a dictionary using SHA256 (e.g. for caching)
    :param dct:
    :return:
    """
    return hashlib.sha256(
        json.dumps(
            dct,
            ensure_ascii=False,
            sort_keys=True,
            indent=None,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
