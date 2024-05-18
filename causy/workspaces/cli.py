import json

import click
import pydantic_yaml
import questionary
import typer
import os

from markdown.extensions.toc import slugify
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str
from jinja2 import (
    Environment,
    select_autoescape,
    ChoiceLoader,
    FileSystemLoader,
    PackageLoader,
)

from causy.graph_model import graph_model_factory
from causy.interfaces import (
    CausyAlgorithmReference,
    CausyAlgorithmReferenceType,
    CausyResult,
)
from causy.serialization import (
    load_algorithm_from_specification,
    load_algorithm_by_reference,
    CausyJSONEncoder,
)
from causy.variables import validate_variable_values, resolve_variables
from causy.workspaces.interfaces import Workspace, Experiment, DataLoader

app = typer.Typer()

WORKSPACE_FILE_NAME = "workspace.yml"

JINJA_ENV = Environment(
    loader=ChoiceLoader(
        [
            PackageLoader("causy", "workspaces/templates"),
            FileSystemLoader("./templates"),
        ]
    ),
    autoescape=select_autoescape(),
)


class WorkspaceNotFoundError(Exception):
    pass


def _current_workspace(fail_if_none: bool = True) -> Workspace:
    """
    Return the current workspace.
    :param fail_if_none: if True, raise an exception if no workspace is found
    :return: the workspace
    """

    workspace_data = None
    workspace_path = os.path.join(os.getcwd(), WORKSPACE_FILE_NAME)
    if os.path.exists(workspace_path):
        with open(workspace_path, "r") as f:
            workspace_data = f.read()

    if fail_if_none and workspace_data is None:
        raise WorkspaceNotFoundError("No workspace found in the current directory")

    workspace = None

    if workspace_data is not None:
        workspace = pydantic_yaml.parse_yaml_raw_as(Workspace, workspace_data)

    return workspace


def _create_pipeline(workspace: Workspace = None) -> Workspace:
    pipeline_creation = questionary.select(
        "Do you want to use an existing pipeline or create a new one?",
        choices=[
            questionary.Choice(
                "Use an existing causy pipeline (preconfigured).", "PRECONFIGURED"
            ),
            questionary.Choice(
                "Eject existing pipeline (allows you to change pipeline configs).",
                "EJECT",
            ),
            questionary.Choice(
                "Create a pipeline skeleton (as a python module).", "SKELETON"
            ),
        ],
    ).ask()

    if pipeline_creation == "PRECONFIGURED":
        from causy.algorithms import AVAILABLE_ALGORITHMS

        pipeline_name = questionary.select(
            "Which pipeline do you want to use?", choices=AVAILABLE_ALGORITHMS.keys()
        ).ask()

        pipeline_reference = AVAILABLE_ALGORITHMS[pipeline_name]
        # make pipeline reference as string
        pipeline = CausyAlgorithmReference(
            reference=pipeline_reference().algorithm.name,
            type=CausyAlgorithmReferenceType.NAME,
        )
        workspace.pipelines[pipeline_name] = pipeline
    elif pipeline_creation == "EJECT":
        from causy.algorithms import AVAILABLE_ALGORITHMS

        pipeline_skeleton = questionary.select(
            "Which pipeline do you want to use?", choices=AVAILABLE_ALGORITHMS.keys()
        ).ask()
        pipeline_reference = AVAILABLE_ALGORITHMS[pipeline_skeleton]
        pipeline_name = questionary.text("Enter the name of the pipeline").ask()
        pipeline_slug = slugify(pipeline_name, "_")
        with open(f"{pipeline_slug}.yml", "w") as f:
            f.write(to_yaml_str(pipeline_reference()._original_algorithm))

        pipeline = CausyAlgorithmReference(
            reference=f"{pipeline_slug}.yml", type=CausyAlgorithmReferenceType.FILE
        )
        workspace.pipelines[pipeline_slug] = pipeline
    elif pipeline_creation == "SKELETON":
        pipeline_name = questionary.text("Enter the name of the pipeline").ask()
        pipeline_slug = slugify(pipeline_name, "_")
        JINJA_ENV.get_template("pipeline.py.tpl").stream(
            pipeline_name=pipeline_name
        ).dump(f"{pipeline_slug}.py")
        pipeline = CausyAlgorithmReference(
            reference=f"{pipeline_slug}.PIPELINE",
            type=CausyAlgorithmReferenceType.PYTHON_MODULE,
        )
        workspace.pipelines[pipeline_slug] = pipeline

    typer.echo(f'Pipeline "{pipeline_name}" created.')

    return workspace


def _create_experiment(workspace: Workspace) -> Workspace:
    experiment_name = questionary.text("Enter the name of the experiment").ask()
    experiment_pipeline = questionary.select(
        "Select the pipeline for the experiment", choices=workspace.pipelines.keys()
    ).ask()
    experiment_data_loader = questionary.select(
        "Select the data loader for the experiment",
        choices=workspace.data_loaders.keys(),
    ).ask()

    experiment_slug = slugify(experiment_name, "_")

    # extract and prefill the variables
    variables = {}
    pipeline = load_algorithm_by_reference(
        workspace.pipelines[experiment_pipeline].type,
        workspace.pipelines[experiment_pipeline].reference,
    )
    if len(pipeline.variables) > 0:
        variables = resolve_variables(pipeline.variables, {})

    workspace.experiments[experiment_slug] = Experiment(
        **{
            "pipeline": experiment_pipeline,
            "data_loader": experiment_data_loader,
            "variables": variables,
        }
    )

    typer.echo(f'Experiment "{experiment_name}" created.')

    return workspace


def _create_data_loader(workspace: Workspace) -> Workspace:
    data_loader_type = questionary.select(
        "Do you want to use an existing pipeline or create a new one?",
        choices=[
            questionary.Choice("Load a JSON File.", "json"),
            questionary.Choice("Load a JSONL File.", "jsonl"),
            questionary.Choice("Load data dynamically (via Python Script).", "dynamic"),
        ],
    ).ask()

    data_loader_name = questionary.text("Enter the name of the data loader").ask()

    if data_loader_type in ["json", "jsonl"]:
        data_loader_path = questionary.path(
            "Choose the file or enter the file name:",
        ).ask()
        data_loader_slug = slugify(data_loader_name, "_")
        workspace.data_loaders[data_loader_slug] = DataLoader(
            **{
                "type": data_loader_type,
                "reference": data_loader_path,
            }
        )
    elif data_loader_type == "dynamic":
        data_loader_slug = slugify(data_loader_name, "_")
        JINJA_ENV.get_template("dataloader.py.tpl").stream(
            data_loader_name=data_loader_name
        ).dump(f"{data_loader_slug}.py")
        workspace.data_loaders[data_loader_slug] = DataLoader(
            **{
                "type": data_loader_type,
                "reference": f"{data_loader_slug}.DataLoader",
            }
        )

    typer.echo(f'Data loader "{data_loader_name}" created.')

    return workspace


def _execute_experiment(workspace: Workspace, experiment: Experiment) -> CausyResult:
    """
    Execute an experiment. This function will load the pipeline and the data loader and execute the pipeline.
    :param workspace:
    :param experiment:
    :return:
    """
    typer.echo(f"Loading Pipeline: {experiment.pipeline}")
    pipeline = load_algorithm_by_reference(
        workspace.pipelines[experiment.pipeline].type,
        workspace.pipelines[experiment.pipeline].reference,
    )

    validate_variable_values(pipeline, experiment.variables)
    variables = resolve_variables(pipeline.variables, experiment.variables)
    typer.echo(f"Using variables: {variables}")

    typer.echo(f"Loading Data: {experiment.data_loader}")
    if workspace.data_loaders[experiment.data_loader].type == "json":
        with open(workspace.data_loaders[experiment.data_loader].reference, "r") as f:
            data = json.load(f)
    elif workspace.data_loaders[experiment.data_loader].type == "jsonl":
        with open(workspace.data_loaders[experiment.data_loader].reference, "r") as f:
            # TODO: use srsly.json_loads
            data = [json.loads(line) for line in f]
    elif workspace.data_loaders[experiment.data_loader].type == "dynamic":
        raise NotImplementedError("Dynamic data loading not implemented yet")

    model = graph_model_factory(pipeline)()
    model.create_graph_from_data(data)
    model.create_all_possible_edges()
    model.execute_pipeline_steps()
    return CausyResult(
        algorithm=workspace.pipelines[experiment.pipeline],
        action_history=model.graph.graph.action_history,
        edges=model.graph.retrieve_edges(),
        nodes=model.graph.nodes,
    )


@app.command()
def create_pipeline():
    """Create a new pipeline in the current workspace."""
    workspace = _current_workspace()
    workspace = _create_pipeline(workspace)

    workspace_path = os.path.join(os.getcwd(), WORKSPACE_FILE_NAME)
    with open(workspace_path, "w") as f:
        f.write(pydantic_yaml.to_yaml_str(workspace))


@app.command()
def create_experiment():
    """Create a new experiment in the current workspace."""
    workspace = _current_workspace()
    workspace = _create_experiment(workspace)

    workspace_path = os.path.join(os.getcwd(), WORKSPACE_FILE_NAME)
    with open(workspace_path, "w") as f:
        f.write(pydantic_yaml.to_yaml_str(workspace))


@app.command()
def create_data_loader():
    """Create a new data loader in the current workspace."""
    workspace = _current_workspace()
    workspace = _create_data_loader(workspace)

    workspace_path = os.path.join(os.getcwd(), WORKSPACE_FILE_NAME)
    with open(workspace_path, "w") as f:
        f.write(pydantic_yaml.to_yaml_str(workspace))


@app.command()
def info():
    """Show general information about the workspace."""
    workspace = _current_workspace()
    typer.echo(f"Workspace: {workspace.name}")
    typer.echo(f"Author: {workspace.author}")
    typer.echo(f"Pipelines: {workspace.pipelines}")
    typer.echo(f"Data loaders: {workspace.data_loaders}")
    typer.echo(f"Experiments: {workspace.experiments}")


@app.command()
def init():
    """
    Initialize a new workspace in the current directory.
    """
    workspace_path = os.path.join(os.getcwd(), WORKSPACE_FILE_NAME)

    if os.path.exists(workspace_path):
        typer.confirm(
            "Workspace already exists. Do you want to overwrite it?", abort=True
        )

    workspace = Workspace(
        **{
            "name": "",
            "author": "",
            "data_loaders": {},
            "pipelines": {},
            "experiments": {},
        }
    )

    current_folder_name = os.path.basename(os.getcwd())

    workspace.name = typer.prompt("Name", default=current_folder_name, type=str)
    workspace.author = typer.prompt(
        "Author", default=os.environ.get("USER", os.environ.get("USERNAME")), type=str
    )

    configure_pipeline = typer.confirm(
        "Do you want to configure a pipeline?", default=False
    )

    workspace.pipelines = {}
    if configure_pipeline:
        workspace = _create_pipeline(workspace)

    configure_data_loader = typer.confirm(
        "Do you want to configure a data loader?", default=False
    )

    workspace.data_loaders = {}
    if configure_data_loader:
        data_loader_type = questionary.select(
            "Do you want to use an existing pipeline or create a new one?",
            choices=[
                questionary.Choice("Load a JSON File.", "json"),
                questionary.Choice("Load a JSONL File.", "jsonl"),
                questionary.Choice(
                    "Load data dynamically (via Python Script).", "dynamic"
                ),
            ],
        ).ask()

        if data_loader_type in ["json", "jsonl"]:
            data_loader_path = questionary.path(
                "Choose the file or enter the file name:",
            ).ask()
            data_loader_name = questionary.text(
                "Enter the name of the data loader"
            ).ask()
            data_loader_slug = slugify(data_loader_name, "_")
            workspace.data_loaders[data_loader_slug] = {
                "type": data_loader_type,
                "reference": data_loader_path,
            }
        elif data_loader_type == "dynamic":
            data_loader_name = questionary.text(
                "Enter the name of the data loader"
            ).ask()
            data_loader_slug = slugify(data_loader_name, "_")
            JINJA_ENV.get_template("dataloader.py.tpl").stream(
                data_loader_name=data_loader_name
            ).dump(f"{data_loader_slug}.py")
            workspace.data_loaders[data_loader_slug] = DataLoader(
                **{
                    "type": data_loader_type,
                    "reference": f"{data_loader_slug}.DataLoader",
                }
            )
    workspace.experiments = {}

    if len(workspace.pipelines) > 0 and len(workspace.data_loaders) > 0:
        configure_experiment = typer.confirm(
            "Do you want to configure an experiment?", default=False
        )

        if configure_experiment:
            workspace = _create_experiment(workspace)

    with open(workspace_path, "w") as f:
        f.write(pydantic_yaml.to_yaml_str(workspace))

    typer.echo(f"Workspace created in {workspace_path}")


@app.command()
def execute(experiment_name=None):
    workspace = _current_workspace()
    if experiment_name is None:
        # execute all experiments
        for experiment_name, experiment in workspace.experiments.items():
            typer.echo(f"Executing experiment: {experiment_name}")
            result = _execute_experiment(workspace, experiment)

            with open(f"{experiment_name}.json", "w") as f:
                f.write(json.dumps(result.model_dump(), cls=CausyJSONEncoder, indent=4))
    else:
        if experiment_name not in workspace.experiments:
            typer.echo(f"Experiment {experiment_name} not found in the workspace.")
            return
        experiment = workspace.experiments[experiment_name]
        typer.echo(f"Executing experiment: {experiment_name}")
        result = _execute_experiment(workspace, experiment)

        with open(f"{experiment_name}.json", "w") as f:
            f.write(json.dumps(result.model_dump(), cls=CausyJSONEncoder, indent=4))
