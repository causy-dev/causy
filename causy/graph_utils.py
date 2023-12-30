import importlib
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def unpack_run(args):
    tst = args[0]
    del args[0]
    return tst(*args)


def generate_to_queue(generator, graph, test_queue):
    print("generate_to_queue")
    for nodes in generator.generate(graph):
        test_queue.put(nodes)


def serialize_module_name(cls):
    return f"{cls.__class__.__module__}.{cls.__class__.__name__}"


def collect_and_execute_tests(
    test_fn,
    test_queue,
    result_queue,
    graph,
    chunk_size_parallel_processing,
    pool,
    generator_process,
):
    """
    Collects tests from the test_queue and executes them.
    :param test_fn:
    :param test_queue:
    :param result_queue:
    :param graph:
    :param chunk_size_parallel_processing:
    :return:
    """

    tests = []
    print("collect_and_execute_tests")
    while not generator_process.ready():
        tests.append(test_queue.get())
        if len(tests) == chunk_size_parallel_processing * 10:
            pool.map_async(
                unpack_run,
                [(test_fn, test, graph, result_queue) for test in tests],
                chunksize=chunk_size_parallel_processing,
            )

    pool.map_async(
        unpack_run,
        [(test_fn, test, graph, result_queue) for test in tests],
        chunksize=chunk_size_parallel_processing,
    )


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
