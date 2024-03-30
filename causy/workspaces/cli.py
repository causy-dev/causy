import click
import pydantic_yaml
import typer
import os

from pydantic_yaml import parse_yaml_raw_as, to_yaml_str

from causy.workspaces.serializer_models import Workspace, Pipeline, PipelineReference

app = typer.Typer()

WORKSPACE_FILE_NAME = "workspace.yml"


def current_workspace(fail_if_none=True):
    """
    Return the current workspace.
    :return:
    """

    # get current path
    # check if there is a workspace
    # return the workspace

    workspace_data = None
    workspace_path = os.path.join(os.getcwd(), WORKSPACE_FILE_NAME)
    if os.path.exists(workspace_path):
        with open(workspace_path, "r") as f:
            workspace_data = f.read()

    if fail_if_none and workspace_data is None:
        raise Exception("No workspace found in the current directory")

    if workspace_data is not None:
        workspace = pydantic_yaml.parse_yaml_raw_as(Workspace, workspace_data)

    return workspace


@app.command()
def main():
    print("Hello World!")


@app.command()
def init():
    """
    Initialize a new workspace in the current directory.
    :retur
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
        use_existing_pipeline = typer.confirm(
            "Do you want to use an existing pipeline?", default=False
        )
        if use_existing_pipeline:
            from causy.algorithms import AVAILABLE_ALGORITHMS

            pipeline_name = click.prompt(
                "\nSelect an tag to deploy: ?",
                type=click.Choice(AVAILABLE_ALGORITHMS.keys()),
            )
            pipeline_reference = AVAILABLE_ALGORITHMS[pipeline_name]
            # make pipeline reference as string
            pipeline = PipelineReference(
                name=pipeline_name, reference=str(pipeline_reference)
            )
            workspace.pipelines[pipeline_name] = pipeline

    workspace.data_loaders = None
    workspace.experiments = None

    with open(workspace_path, "w") as f:
        f.write(pydantic_yaml.to_yaml_str(workspace))

    print(f"Workspace created in {workspace_path}")


@app.commanda()
def execute(experiment_name=""):
    pass
