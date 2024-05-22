import typer
from causy.serialization import load_json
from causy.ui.server import server
from causy.workspaces.cli import _current_workspace


def ui(result_file: str = None):
    """Start the causy UI."""
    if not result_file:
        workspace = _current_workspace()
        server_config, server_runner = server(workspace=workspace)
    else:
        result = load_json(result_file)
        server_config, server_runner = server(result=result)

    typer.launch(f"http://{server_config.host}:{server_config.port}")
    typer.echo(f"🚀 Starting server at http://{server_config.host}:{server_config.port}")
    server_runner.run()
