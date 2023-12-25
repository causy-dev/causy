import importlib
from typing import List, Tuple


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

    if "params" not in step.keys():
        return st_function()
    else:
        return st_function(**step["params"])


def load_pipeline_steps_by_definition(steps):
    pipeline = []
    for step in steps:
        st_function = load_pipeline_artefact_by_definition(step)
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
