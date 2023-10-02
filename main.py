import importlib
import json
from json import JSONEncoder

import typer

from graph import graph_model_factory

app = typer.Typer()
import logging


def load_json(pipeline_file: str):
    with open(pipeline_file, "r") as file:
        pipeline = json.loads(file.read())
    return pipeline


def create_pipeline(pipeline_config: dict):
    pipeline = []
    for step in pipeline_config["steps"]:
        path = ".".join(step["step"].split(".")[:-1])
        cls = step["step"].split(".")[-1]
        st_function = importlib.import_module(path)
        st_function = getattr(st_function, cls)
        if "params" not in step.keys():
            pipeline.append(st_function())
        else:
            pipeline.append(st_function(**step["params"]))

    return pipeline


def show_edges(graph):
    for u in graph.edges:
        for v in graph.edges[u]:
            print(f"{u.name} -> {v.name}: {graph.edges[u][v]}")


class MyJSONEncoder(JSONEncoder):
    def default(self, obj):
        return obj.to_dict()


@app.command()
def execute(
    pipeline_file: str,
    data_file: str,
    graph_actions_save_file: str = None,
    log_level: str = "ERROR",
):
    typer.echo(f"💾 Loading pipeline from {pipeline_file}")
    pipeline_config = load_json(pipeline_file)
    # set log level
    logging.basicConfig(level=log_level)
    pipeline = create_pipeline(pipeline_config)
    model = graph_model_factory(pipeline_steps=pipeline)()

    model.create_graph_from_data(load_json(data_file))
    model.create_all_possible_edges()
    typer.echo("🕵🏻‍♀  Executing pipeline steps...")
    model.execute_pipeline_steps()
    show_edges(model.graph)
    if graph_actions_save_file:
        typer.echo(f"💾 Saving graph actions to {graph_actions_save_file}")
        with open(graph_actions_save_file, "w") as file:
            file.write(
                json.dumps(model.graph.action_history, cls=MyJSONEncoder, indent=4)
            )
        # model.save_graph_actions(graph_actions_save_file)


@app.command()
def visualize(output: str):
    raise NotImplementedError()


if __name__ == "__main__":
    app()
