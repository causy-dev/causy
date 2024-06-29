import json
import logging

import typer

from causy.data_loader import JSONDataLoader
from causy.graph_model import graph_model_factory
from causy.models import (
    Result,
    AlgorithmReferenceType,
)
from causy.serialization import (
    serialize_algorithm,
    load_algorithm_from_specification,
    CausyJSONEncoder,
    load_json,
)
from causy.graph_utils import (
    retrieve_edges,
)
from causy.ui.cli import ui as ui_app
from causy.workspaces.cli import workspace_app as workspaces_app
from causy.causal_discovery import AVAILABLE_ALGORITHMS

app = typer.Typer()

app.add_typer(workspaces_app, name="workspace")
app.command(name="ui", help="run causy ui")(ui_app)


@app.command()
def eject(algorithm: str, output_file: str):
    logging.warning(
        f"Ejecting pipelines outside of workspace context is deprecated. Please use workspaces instead."
    )

    typer.echo(f"💾 Loading algorithm {algorithm}")
    model = AVAILABLE_ALGORITHMS[algorithm]()
    result = serialize_algorithm(model, algorithm_name=algorithm)
    typer.echo(f"💾 Saving algorithm {algorithm} to {output_file}")
    with open(output_file, "w") as file:
        file.write(json.dumps(result, indent=4))


@app.command()
def execute(
    data_file: str,
    pipeline: str = None,
    algorithm: str = None,
    output_file: str = None,
    log_level: str = "ERROR",
):
    logging.warning(
        f"Executing outside of workspaces is deprecated and will be removed in future versions. Please use workspaces instead."
    )
    logging.basicConfig(level=log_level)
    if pipeline:
        typer.echo(f"💾 Loading pipeline from {pipeline}")
        model_dict = load_json(pipeline)
        algorithm = load_algorithm_from_specification(model_dict)
        model = graph_model_factory(algorithm=algorithm)()
        algorithm_reference = {
            "type": AlgorithmReferenceType.FILE,
            "reference": pipeline,  # TODO: how to reference pipeline in a way that it can be loaded?
        }
    elif algorithm:
        typer.echo(f"💾 Creating pipeline from algorithm {algorithm}")
        model = AVAILABLE_ALGORITHMS[algorithm]()
        algorithm_reference = {
            "type": AlgorithmReferenceType.NAME,
            "reference": algorithm,
        }

    else:
        raise ValueError("Either pipeline_file or algorithm must be specified")

    dl = JSONDataLoader(data_file)
    # initialize from json
    model.create_graph_from_data(dl)

    # TODO: I should become a configurable skeleton builder
    model.create_all_possible_edges()

    typer.echo("🕵🏻‍♀  Executing pipeline steps...")
    model.execute_pipeline_steps()
    edges = []
    for edge in retrieve_edges(model.graph):
        print(
            f"{model.graph.nodes[edge[0]].name} -> {model.graph.nodes[edge[1]].name}: {model.graph.edges[edge[0]][edge[1]]}"
        )

    result = Result(
        algorithm=algorithm_reference,
        action_history=model.graph.graph.action_history,
        edges=model.graph.retrieve_edges(),
        nodes=model.graph.nodes,
    )

    if output_file:
        typer.echo(f"💾 Saving graph actions to {output_file}")
        with open(output_file, "w") as file:
            file.write(json.dumps(result.model_dump(), cls=CausyJSONEncoder, indent=4))


if __name__ == "__main__":
    app()
