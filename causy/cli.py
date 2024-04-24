import importlib
import json
from datetime import datetime
from json import JSONEncoder
import logging

import typer

from causy.graph_model import graph_model_factory
from causy.serialization import serialize_algorithm, load_algorithm_from_specification
from causy.graph_utils import (
    retrieve_edges,
)
from causy.ui import server
from causy.workspaces.cli import app as workspaces_app

app = typer.Typer()

app.add_typer(workspaces_app, name="workspace")


def load_json(pipeline_file: str):
    with open(pipeline_file, "r") as file:
        pipeline = json.loads(file.read())
    return pipeline


def load_algorithm_from_reference(algorithm: str):
    st_function = importlib.import_module("causy.algorithms")
    st_function = getattr(st_function, algorithm)
    if not st_function:
        raise ValueError(f"Algorithm {algorithm} not found")
    return st_function


class MyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "serialize"):
            return obj.serialize()
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return None


@app.command()
def eject(algorithm: str, output_file: str):
    typer.echo(f"💾 Loading algorithm {algorithm}")
    model = load_algorithm_from_reference(algorithm)()
    result = serialize_algorithm(model, algorithm_name=algorithm)
    typer.echo(f"💾 Saving algorithm {algorithm} to {output_file}")
    with open(output_file, "w") as file:
        file.write(json.dumps(result, indent=4))


@app.command()
def ui(result_file: str):
    result = load_json(result_file)

    server_config, server_runner = server(result)
    typer.launch(f"http://{server_config.host}:{server_config.port}")
    typer.echo(f"🚀 Starting server at http://{server_config.host}:{server_config.port}")
    server_runner.run()


@app.command()
def execute(
    data_file: str,
    pipeline: str = None,
    algorithm: str = None,
    output_file: str = None,
    log_level: str = "ERROR",
):
    logging.basicConfig(level=log_level)
    if pipeline:
        typer.echo(f"💾 Loading pipeline from {pipeline}")
        model_dict = load_json(pipeline)
        algorithm = load_algorithm_from_specification(model_dict)
        model = graph_model_factory(algorithm=algorithm)()
        algorithm_reference = {
            "type": "pipeline",
            "reference": pipeline,  # TODO: how to reference pipeline in a way that it can be loaded?
        }
    elif algorithm:
        typer.echo(f"💾 Creating pipeline from algorithm {algorithm}")
        model = load_algorithm_from_reference(algorithm)()
        algorithm_reference = {
            "type": "default",
            "reference": algorithm,
        }

    else:
        raise ValueError("Either pipeline_file or algorithm must be specified")

    # initialize from json
    model.create_graph_from_data(load_json(data_file))

    # TODO: I should become a configurable skeleton builder
    model.create_all_possible_edges()

    typer.echo("🕵🏻‍♀  Executing pipeline steps...")
    model.execute_pipeline_steps()
    edges = []
    for edge in retrieve_edges(model.graph):
        print(
            f"{model.graph.nodes[edge[0]].name} -> {model.graph.nodes[edge[1]].name}: {model.graph.edges[edge[0]][edge[1]]}"
        )

        edge_value = model.graph.edges[edge[0]][edge[1]].model_dump()

        if "edge_type" in edge_value:
            edge_value["edge_type"] = edge_value["edge_type"]["name"]

        if "u" in edge_value:
            del edge_value["u"]
            del edge_value["v"]

        edges.append(
            {
                "from": model.graph.nodes[edge[0]].serialize(),
                "to": model.graph.nodes[edge[1]].serialize(),
                "value": edge_value,
            }
        )

    if output_file:
        typer.echo(f"💾 Saving graph actions to {output_file}")
        with open(output_file, "w") as file:
            export = {
                "name": algorithm,
                "created_at": datetime.now().isoformat(),
                "algorithm": algorithm_reference,
                "steps": model.graph.action_history,
                "nodes": model.graph.nodes,
                "edges": edges,
            }
            file.write(json.dumps(export, cls=MyJSONEncoder, indent=4))


if __name__ == "__main__":
    app()
