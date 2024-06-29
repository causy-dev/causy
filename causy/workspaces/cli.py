import json
import logging
from collections import OrderedDict
from datetime import datetime
from typing import List, Dict

import pydantic_yaml
import questionary
import typer
import os
import sys

import yaml
from markdown.extensions.toc import slugify
from pydantic_yaml import to_yaml_str
from jinja2 import (
    Environment,
    select_autoescape,
    ChoiceLoader,
    FileSystemLoader,
    PackageLoader,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from causy.graph_model import graph_model_factory
from causy.graph_utils import hash_dictionary
from causy.models import (
    AlgorithmReference,
    AlgorithmReferenceType,
    Result,
)
from causy.serialization import (
    load_algorithm_by_reference,
    CausyJSONEncoder,
    deserialize_result,
)
from causy.variables import validate_variable_values, resolve_variables
from causy.workspaces.models import Workspace, Experiment
from causy.data_loader import DataLoaderReference, load_data_loader

workspace_app = typer.Typer()

pipeline_app = typer.Typer()
experiment_app = typer.Typer()
dataloader_app = typer.Typer()
logger = logging.getLogger(__name__)

NO_COLOR = os.environ.get("NO_COLOR", False)

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


def show_error(message: str):
    if NO_COLOR:
        typer.echo(f"❌ {message}", err=True)
    else:
        typer.echo(typer.style(f"❌ {message}", fg=typer.colors.RED), err=True)


def show_success(message: str):
    typer.echo(f"✅ {message}")


def write_to_workspace(workspace: Workspace):
    workspace_path = os.path.join(os.getcwd(), WORKSPACE_FILE_NAME)
    ordered_val = OrderedDict(json.loads(workspace.model_dump_json()))
    yaml.add_representer(
        OrderedDict,
        lambda dumper, data: dumper.represent_mapping(
            "tag:yaml.org,2002:map", data.items()
        ),
    )
    output = yaml.dump(ordered_val)
    with open(workspace_path, "w") as f:
        f.write(output)


def _current_workspace(fail_if_none: bool = True) -> Workspace:
    """
    Return the current workspace.
    :param fail_if_none: if True, raise an exception if no workspace is found
    :return: the workspace
    """

    workspace_data = None
    workspace_path = os.path.join(os.getcwd(), WORKSPACE_FILE_NAME)
    # add the current directory to the path to allow for imports
    sys.path.append(os.getcwd())
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
        from causy.causal_discovery.constraint.algorithms import AVAILABLE_ALGORITHMS

        pipeline_reference = questionary.select(
            "Which pipeline do you want to use?", choices=AVAILABLE_ALGORITHMS.keys()
        ).ask()
        pipeline_reference = AVAILABLE_ALGORITHMS[pipeline_reference]
        # make pipeline reference as string
        pipeline = AlgorithmReference(
            reference=pipeline_reference().algorithm.name,
            type=AlgorithmReferenceType.NAME,
        )

        pipeline_name = questionary.text("Enter the name of the pipeline").ask()
        pipeline_name = slugify(pipeline_name, "_")

        workspace.pipelines[pipeline_name] = pipeline
    elif pipeline_creation == "EJECT":
        from causy.causal_discovery.constraint.algorithms import AVAILABLE_ALGORITHMS

        pipeline_skeleton = questionary.select(
            "Which pipeline do you want to use?", choices=AVAILABLE_ALGORITHMS.keys()
        ).ask()
        pipeline_reference = AVAILABLE_ALGORITHMS[pipeline_skeleton]
        pipeline_name = questionary.text("Enter the name of the pipeline").ask()
        pipeline_slug = slugify(pipeline_name, "_")
        with open(f"{pipeline_slug}.yml", "w") as f:
            f.write(to_yaml_str(pipeline_reference()._original_algorithm))

        pipeline = AlgorithmReference(
            reference=f"{pipeline_slug}.yml", type=AlgorithmReferenceType.FILE
        )

        workspace.pipelines[pipeline_name] = pipeline
    elif pipeline_creation == "SKELETON":
        pipeline_name = questionary.text("Enter the name of the pipeline").ask()
        pipeline_slug = slugify(pipeline_name, "_")
        JINJA_ENV.get_template("pipeline.py.tpl").stream(
            pipeline_name=pipeline_name
        ).dump(f"{pipeline_slug}.py")
        pipeline = AlgorithmReference(
            reference=f"{pipeline_slug}.PIPELINE",
            type=AlgorithmReferenceType.PYTHON_MODULE,
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
        choices=workspace.dataloaders.keys(),
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
            "dataloader": experiment_data_loader,
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
        workspace.dataloaders[data_loader_slug] = DataLoaderReference(
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
        workspace.dataloaders[data_loader_slug] = DataLoaderReference(
            **{
                "type": data_loader_type,
                "reference": f"{data_loader_slug}.DataLoader",
            }
        )

    typer.echo(f'Data loader "{data_loader_name}" created.')

    return workspace


def _execute_experiment(workspace: Workspace, experiment: Experiment) -> Result:
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

    typer.echo(f"Loading Data: {experiment.dataloader}")
    data_loader = load_data_loader(workspace.dataloaders[experiment.dataloader])
    model = graph_model_factory(pipeline, experiment.variables)()
    model.create_graph_from_data(data_loader)
    model.create_all_possible_edges()
    task_count = len(model.pipeline_steps)
    current = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        prev_task = None
        prev_task_data = None
        for task in model.execute_pipeline_step_with_progress():
            current += 1
            if prev_task is not None:
                progress.update(
                    prev_task,
                    completed=True,
                    current=1,
                    description=f"✅ {prev_task_data['step']} ({round(task['previous_duration'])}s)",
                )
            prev_task = progress.add_task(description=task["step"], total=1)
            prev_task_data = task

    return Result(
        algorithm=workspace.pipelines[experiment.pipeline],
        action_history=model.graph.graph.action_history,
        edges=model.graph.retrieve_edges(),
        nodes=model.graph.nodes,
        variables=variables,
        data_loader_hash=data_loader.hash(),
        algorithm_hash=pipeline.hash(),
        variables_hash=hash_dictionary(variables),
    )


def _load_latest_experiment_result(
    workspace: Workspace, experiment_name: str
) -> Experiment:
    versions = _load_experiment_versions(workspace, experiment_name)

    if experiment_name not in workspace.experiments:
        raise ValueError(f"Experiment {experiment_name} not found in the workspace")

    if len(versions) == 0:
        raise ValueError(f"Experiment {experiment_name} not found in the file system")

    with open(f"{experiment_name}_{versions[0]}.json", "r") as f:
        experiment = json.load(f)

    return experiment


def _load_experiment_result(
    workspace: Workspace, experiment_name: str, version_number: int
) -> Dict[str, any]:
    if experiment_name not in workspace.experiments:
        raise ValueError(f"Experiment {experiment_name} not found in the workspace")

    if version_number not in _load_experiment_versions(workspace, experiment_name):
        raise ValueError(
            f"Version {version_number} not found for experiment {experiment_name}"
        )

    with open(f"{experiment_name}_{version_number}.json", "r") as f:
        experiment = json.load(f)

    return experiment


def _load_experiment_versions(workspace: Workspace, experiment_name: str) -> List[int]:
    versions = []
    for file in os.listdir():
        # check for files if they have the right prefix followed by a unix timestamp (int) and the file extension, e.g. experiment_123456789.json.
        # Extract the unix timestamp
        if file.startswith(f"{experiment_name}_") and file.endswith(".json"):
            segments = file.split("_")
            timestamp = int(segments[-1].split(".")[0])
            name = "_".join(segments[:-1])
            if name != experiment_name:
                # an experiment with a different name
                continue
            versions.append(timestamp)
    return sorted(versions, reverse=True)


def _save_experiment_result(workspace: Workspace, experiment_name: str, result: Result):
    timestamp = int(datetime.timestamp(result.created_at))
    with open(f"{experiment_name}_{timestamp}.json", "w") as f:
        f.write(json.dumps(result.model_dump(), cls=CausyJSONEncoder, indent=4))


@pipeline_app.command(name="add")
def create_pipeline():
    """Create a new pipeline in the current workspace."""
    workspace = _current_workspace()
    workspace = _create_pipeline(workspace)

    write_to_workspace(workspace)


@pipeline_app.command(name="rm")
def remove_pipeline(pipeline_name: str):
    """Remove a pipeline from the current workspace."""
    workspace = _current_workspace()

    if pipeline_name not in workspace.pipelines:
        show_error(f"Pipeline {pipeline_name} not found in the workspace.")
        return

    # check if the pipeline is still in use
    for experiment_name, experiment in workspace.experiments.items():
        if experiment.pipeline == pipeline_name:
            show_error(
                f"Pipeline {pipeline_name} is still in use by experiment {experiment_name}. Cannot remove."
            )
            return

    del workspace.pipelines[pipeline_name]

    write_to_workspace(workspace)

    show_success(f"Pipeline {pipeline_name} removed from the workspace.")


def _experiment_needs_reexecution(workspace: Workspace, experiment_name: str) -> bool:
    """
    Check if an experiment needs to be re-executed.
    :param workspace:
    :param experiment_name:
    :return:
    """
    if experiment_name not in workspace.experiments:
        raise ValueError(f"Experiment {experiment_name} not found in the workspace")

    versions = _load_experiment_versions(workspace, experiment_name)

    if len(versions) == 0:
        logger.info(f"Experiment {experiment_name} not found in the file system.")
        return True

    latest_experiment = _load_latest_experiment_result(workspace, experiment_name)
    experiment = workspace.experiments[experiment_name]
    latest_experiment = deserialize_result(latest_experiment)
    if (
        latest_experiment.algorithm_hash is None
        or latest_experiment.data_loader_hash is None
    ):
        logger.info(f"Experiment {experiment_name} has no hashes.")
        return True

    pipeline = load_algorithm_by_reference(
        workspace.pipelines[experiment.pipeline].type,
        workspace.pipelines[experiment.pipeline].reference,
    )

    validate_variable_values(pipeline, experiment.variables)
    variables = resolve_variables(pipeline.variables, experiment.variables)

    if latest_experiment.variables_hash != hash_dictionary(variables):
        logger.info(f"Experiment {experiment_name} has different variables.")
        return True

    model = graph_model_factory(pipeline, variables)()
    if latest_experiment.algorithm_hash != model.algorithm.hash():
        logger.info(f"Experiment {experiment_name} has a different pipeline.")
        return True

    data_loder = load_data_loader(workspace.dataloaders[experiment.dataloader])
    if latest_experiment.data_loader_hash != data_loder.hash():
        logger.info(
            f"Experiment {experiment_name} has a different data loader/dataset."
        )
        return True

    return False


def _clear_experiment(experiment_name: str, workspace: Workspace):
    versions = _load_experiment_versions(workspace, experiment_name)
    versions_removed = 0
    for version in versions:
        try:
            os.remove(f"{experiment_name}_{version}.json")
            versions_removed += 1
        except FileNotFoundError:
            pass
    return versions_removed


@experiment_app.command(name="add")
def create_experiment():
    """Create a new experiment in the current workspace."""
    workspace = _current_workspace()
    workspace = _create_experiment(workspace)

    write_to_workspace(workspace)


@experiment_app.command(name="rm")
def remove_experiment(experiment_name: str):
    """Remove an experiment from the current workspace."""
    workspace = _current_workspace()

    if experiment_name not in workspace.experiments:
        show_error(f"Experiment {experiment_name} not found in the workspace.")
        return

    versions_removed = _clear_experiment(experiment_name, workspace)

    del workspace.experiments[experiment_name]

    write_to_workspace(workspace)

    show_success(
        f"Experiment {experiment_name} removed from the workspace. Removed {versions_removed} versions."
    )


@experiment_app.command(name="clear")
def clear_experiment(experiment_name: str):
    """Clear all versions of an experiment."""
    workspace = _current_workspace()

    if experiment_name not in workspace.experiments:
        show_error(f"Experiment {experiment_name} not found in the workspace.")
        return

    versions_removed = _clear_experiment(experiment_name, workspace)

    write_to_workspace(workspace)

    show_success(
        f"Experiment {experiment_name} cleared. Removed {versions_removed} versions."
    )


@experiment_app.command(name="update-variable")
def update_experiment_variable(
    experiment_name: str, variable_name: str, variable_value: str
):
    """Update a variable in an experiment."""
    workspace = _current_workspace()

    if experiment_name not in workspace.experiments:
        show_error(f"Experiment {experiment_name} not found in the workspace.")
        return

    experiment = workspace.experiments[experiment_name]

    pipeline = load_algorithm_by_reference(
        workspace.pipelines[experiment.pipeline].type,
        workspace.pipelines[experiment.pipeline].reference,
    )

    current_variable = None
    for existing_variable in pipeline.variables:
        if variable_name == existing_variable.name:
            current_variable = existing_variable
            break
    else:
        show_error(f"Variable {variable_name} not found in the experiment.")
        return

    # try to cast the variable value to the correct type
    try:
        variable_value = current_variable._PYTHON_TYPE(variable_value)
    except ValueError:
        show_error(
            f'Variable {variable_name} should be {current_variable.type}. But got "{variable_value}" which is not a valid value.'
        )
        return

    # check if the variable is a valid value
    if not validate_variable_values(pipeline, {variable_name: variable_value}):
        show_error(f"Variable {variable_name} is not a valid value.")
        return

    experiment.variables[variable_name] = variable_value

    write_to_workspace(workspace)

    show_success(f"Variable {variable_name} updated in experiment {experiment_name}.")


@dataloader_app.command(name="add")
def create_data_loader():
    """Create a new data loader in the current workspace."""
    workspace = _current_workspace()
    workspace = _create_data_loader(workspace)

    write_to_workspace(workspace)


@dataloader_app.command(name="rm")
def remove_data_loader(data_loader_name: str):
    """Remove a data loader from the current workspace."""
    workspace = _current_workspace()

    if data_loader_name not in workspace.dataloaders:
        show_error(f"Data loader {data_loader_name} not found in the workspace.")
        return

    # check if the data loader is still in use
    for experiment_name, experiment in workspace.experiments.items():
        if experiment.dataloader == data_loader_name:
            show_error(
                f"Data loader {data_loader_name} is still in use by experiment {experiment_name}. Cannot remove."
            )
            return

    del workspace.dataloaders[data_loader_name]

    write_to_workspace(workspace)

    show_success(f"Data loader {data_loader_name} removed from the workspace.")


@workspace_app.command()
def info():
    """Show general information about the workspace."""
    workspace = _current_workspace()
    typer.echo(f"Workspace: {workspace.name}")
    typer.echo(f"Author: {workspace.author}")
    typer.echo(f"Pipelines: {workspace.pipelines}")
    typer.echo(f"Data loaders: {workspace.dataloaders}")
    typer.echo(f"Experiments: {workspace.experiments}")


@workspace_app.command()
def init():
    """
    Initialize a new workspace in the current directory.
    """
    workspace_path = os.path.join(os.getcwd(), WORKSPACE_FILE_NAME)
    sys.path.append(os.getcwd())

    if os.path.exists(workspace_path):
        typer.confirm(
            "Workspace already exists. Do you want to overwrite it?", abort=True
        )

    workspace = Workspace(
        **{
            "name": "",
            "author": "",
            "dataloaders": {},
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

    workspace.dataloaders = {}
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
            workspace.dataloaders[data_loader_slug] = DataLoaderReference(
                **{
                    "type": data_loader_type,
                    "reference": data_loader_path,
                }
            )
        elif data_loader_type == "dynamic":
            data_loader_name = questionary.text(
                "Enter the name of the data loader"
            ).ask()
            data_loader_slug = slugify(data_loader_name, "_")
            JINJA_ENV.get_template("dataloader.py.tpl").stream(
                data_loader_name=data_loader_name
            ).dump(f"{data_loader_slug}.py")
            workspace.dataloaders[data_loader_slug] = DataLoaderReference(
                **{
                    "type": data_loader_type,
                    "reference": f"{data_loader_slug}.DataLoader",
                }
            )
    workspace.experiments = {}

    if len(workspace.pipelines) > 0 and len(workspace.dataloaders) > 0:
        configure_experiment = typer.confirm(
            "Do you want to configure an experiment?", default=False
        )

        if configure_experiment:
            workspace = _create_experiment(workspace)

    with open(workspace_path, "w") as f:
        f.write(pydantic_yaml.to_yaml_str(workspace))

    typer.echo(f"Workspace created in {workspace_path}")


@workspace_app.command()
def execute(experiment_name: str = None, force_reexecution: bool = False):
    """
    Execute an experiment or all experiments in the workspace.
    """
    workspace = _current_workspace()
    if experiment_name is None:
        # execute all experiments
        for experiment_name, experiment in workspace.experiments.items():
            try:
                needs_reexecution = _experiment_needs_reexecution(
                    workspace, experiment_name
                )
            except ValueError as e:
                show_error(str(e))
                needs_reexecution = True

            if needs_reexecution is False and force_reexecution is False:
                typer.echo(f"Skipping experiment: {experiment_name}. (no changes)")
                continue
            typer.echo(f"Executing experiment: {experiment_name}")
            result = _execute_experiment(workspace, experiment)
            _save_experiment_result(workspace, experiment_name, result)
    else:
        if experiment_name not in workspace.experiments:
            typer.echo(f"Experiment {experiment_name} not found in the workspace.")
            return
        experiment = workspace.experiments[experiment_name]
        typer.echo(f"Executing experiment: {experiment_name}")
        result = _execute_experiment(workspace, experiment)

        _save_experiment_result(workspace, experiment_name, result)


@workspace_app.command()
def diff(experiment_names: List[str], only_differences: bool = False):
    """
    Show the differences between multiple experiment results.
    """
    workspace = _current_workspace()
    if len(experiment_names) < 2:
        show_error("Please provide at least two experiment names/versions.")
        return

    experiments_to_compare = []
    resolved_experiments = []

    # check if the experiment strings are experiments or experiment_versions and load the respective experiments/versions
    for experiment_name in experiment_names:
        if experiment_name not in workspace.experiments:
            potential_version = experiment_name.split("_")[-1]
            try:
                version = int(potential_version)
            except ValueError:
                show_error(f"Experiment {experiment_name} not found in the workspace")
                return
            experiment_name = "_".join(experiment_name.split("_")[:-1])
            if version not in _load_experiment_versions(workspace, experiment_name):
                show_error(
                    f"Version {version} not found for experiment {experiment_name}"
                )
                return
            experiment_result = deserialize_result(
                _load_experiment_result(workspace, experiment_name, version)
            )
            experiment = workspace.experiments[experiment_name]
            experiment_version = f"{experiment_name}_{version}"

        else:
            experiment = workspace.experiments[experiment_name]
            experiment_result = deserialize_result(
                _load_latest_experiment_result(workspace, experiment_name)
            )
            experiment_version = f"{experiment_name}_latest"

        experiments_to_compare.append(
            {
                "result": experiment_result,
                "experiment": experiment,
                "version": experiment_version,
            }
        )
        resolved_experiments.append(experiment_version)
    find_equivalents = {}

    # find the differences between all experiments and the differences for each of the other edges

    for experiment in experiments_to_compare:
        for edge in experiment["result"].edges:
            u, v = sorted([edge.u.name, edge.v.name])
            if u not in find_equivalents:
                find_equivalents[u] = {}
            if v not in find_equivalents[u]:
                find_equivalents[u][v] = {}

            if experiment["version"] in find_equivalents[u][v]:
                if find_equivalents[u][v][experiment["version"]] != edge:
                    typer.echo(
                        f"Experiment {experiment['experiment']} has an inconsistent edge {u} -> {v}"
                    )
            else:
                find_equivalents[u][v][experiment["version"]] = edge

    experiment_table = []

    for node_u, s in find_equivalents.items():
        for node_v, result in s.items():
            experiment_table_row = {exp: None for exp in resolved_experiments}
            for experiment, edge in result.items():
                experiment_table_row[experiment] = edge

            experiment_table.append(experiment_table_row)

    table = Table()
    table.add_column("Edge")

    for experiment in resolved_experiments:
        table.add_column(experiment, justify="center")

    for row in experiment_table:
        elements = [key for key in row.values()]
        first_element = None
        for e in elements:
            if e is not None:
                first_element = e
                break

        all_elements_same = all([e == first_element for e in elements])
        if only_differences and all_elements_same:
            continue
        table.add_row(
            *[f"{first_element.u.name} - {first_element.v.name}"]
            + [
                f"{row[experiment].edge_type.STR_REPRESENTATION}"
                if row[experiment]
                else ""
                for experiment in resolved_experiments
            ],
            style="green" if all_elements_same else "red",
        )

    console = Console()
    console.print(table)


workspace_app.add_typer(pipeline_app, name="pipeline", help="Manage pipelines")
workspace_app.add_typer(experiment_app, name="experiment", help="Manage experiments")
workspace_app.add_typer(dataloader_app, name="dataloader", help="Manage data loaders")
