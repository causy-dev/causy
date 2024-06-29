import logging
import os
from datetime import datetime
from importlib.metadata import version
from typing import Dict, Any, List, Optional

import fastapi
import typer
import uvicorn

from causy.models import (
    AlgorithmReference,
    Algorithm,
    AlgorithmReferenceType,
)
from causy.serialization import load_algorithm_by_reference
from causy.ui.models import ExtendedResult, ExtendedExperiment, ExperimentVersion
from causy.workspaces.cli import (
    _load_latest_experiment_result,
    _load_experiment_versions,
    _load_experiment_result,
)
from causy.workspaces.models import Workspace
from fastapi import APIRouter, HTTPException
from starlette.staticfiles import StaticFiles

logger = logging.getLogger(__name__)
API_ROUTES = APIRouter()
MODEL: Optional[ExtendedResult] = None
WORKSPACE: Optional[Workspace] = None


@API_ROUTES.get("/status", response_model=Dict[str, Any])
async def get_status():
    """Get the current status of the API."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "workspace_loaded": WORKSPACE is not None,
        "mode": "workspace" if WORKSPACE else "model",
        "causy_version": version("causy"),
    }


@API_ROUTES.get("/model", response_model=ExtendedResult)
async def get_model():
    """Get the current model."""
    if not MODEL:
        raise HTTPException(404, "No model loaded")
    return MODEL


@API_ROUTES.get("/workspace", response_model=Workspace)
async def get_workspace():
    if not WORKSPACE:
        raise HTTPException(404, "No workspace loaded")
    return WORKSPACE


@API_ROUTES.get("/experiments/{experiment_name}/latest", response_model=ExtendedResult)
async def get_latest_experiment(experiment_name: str):
    """Get the current experiment."""
    if not WORKSPACE:
        raise HTTPException(404, "No workspace loaded")

    if experiment_name not in WORKSPACE.experiments:
        raise HTTPException(404, "Experiment not found")

    try:
        experiment = _load_latest_experiment_result(WORKSPACE, experiment_name)
    except Exception as e:
        raise HTTPException(400, str(e))

    version = _load_experiment_versions(WORKSPACE, experiment_name)[0]

    experiment["algorithm"] = AlgorithmReference(**experiment["algorithm"])
    experiment["version"] = version
    experiment = ExtendedResult(**experiment)

    return experiment


@API_ROUTES.get(
    "/experiments/{experiment_name}/{version_number}",
    response_model=ExtendedResult,
)
async def get_experiment(experiment_name: str, version_number: int):
    """Get the current experiment."""
    if not WORKSPACE:
        raise HTTPException(404, "No workspace loaded")

    if experiment_name not in WORKSPACE.experiments:
        raise HTTPException(404, "Experiment not found")

    try:
        experiment = _load_experiment_result(WORKSPACE, experiment_name, version_number)
    except Exception as e:
        raise HTTPException(400, str(e))

    experiment["algorithm"] = AlgorithmReference(**experiment["algorithm"])
    experiment["version"] = version_number
    experiment = ExtendedResult(**experiment)

    return experiment


@API_ROUTES.get("/experiments", response_model=List[ExtendedExperiment])
async def get_experiments():
    """Get the current experiment."""
    if not WORKSPACE:
        raise HTTPException(404, "No workspace loaded")

    experiments = []
    for experiment_name, experiment in WORKSPACE.experiments.items():
        extended_experiment = ExtendedExperiment(**experiment.model_dump())
        versions = []
        for experiment_version in _load_experiment_versions(WORKSPACE, experiment_name):
            versions.append(
                ExperimentVersion(
                    version=experiment_version,
                    name=datetime.fromtimestamp(experiment_version).isoformat(),
                )
            )
        extended_experiment.versions = versions
        extended_experiment.name = experiment_name
        experiments.append(extended_experiment)

    return experiments


@API_ROUTES.get("/algorithm/{reference_type}/{reference}", response_model=Algorithm)
async def get_algorithm(reference_type: str, reference: str):
    """Get the current algorithm."""
    if reference.startswith("/") or ".." in reference:
        raise HTTPException(400, "Invalid reference")

    if reference_type not in AlgorithmReferenceType.__members__.values():
        raise HTTPException(400, "Invalid reference type")

    try:
        algorithm = load_algorithm_by_reference(reference_type, reference)
        return algorithm
    except Exception as e:
        raise HTTPException(400, str(e))


def _create_ui_app(with_static=True):
    """Get the server."""
    app = fastapi.FastAPI(
        title="causy-api",
        version=version("causy"),
        description="causys internal api to serve data from the result files",
    )
    app.include_router(API_ROUTES, prefix="/api/v1", tags=["api"])
    if with_static:
        app.mount(
            "",
            StaticFiles(
                directory=os.path.join(os.path.dirname(__file__), "..", "static"),
                html=True,
            ),
            name="static",
        )
    return app


def _set_model(result: Dict[str, Any]):
    """Set the model."""
    global MODEL
    # for testing
    if result is None:
        MODEL = None
        return

    result["algorithm"] = AlgorithmReference(**result["algorithm"])
    MODEL = ExtendedResult(**result)


def _set_workspace(workspace: Workspace):
    """Set the workspace."""
    global WORKSPACE
    WORKSPACE = workspace


def is_port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def server(result: Dict[str, Any] = None, workspace: Workspace = None):
    """Create the FastAPI server."""
    app = _create_ui_app()

    if result:
        _set_model(result)

    if workspace:
        _set_workspace(workspace)

    if not MODEL and not WORKSPACE:
        raise ValueError("No model or workspace provided")

    host = os.getenv("HOST", "localhost")
    is_port_from_env = os.getenv("PORT")
    if is_port_from_env:
        port = int(is_port_from_env)
    else:
        port = int(os.getenv("PORT", "8000"))
        while is_port_in_use(host, port):
            port += 1
            if port > 65535:
                raise ValueError("No free port available")

    cors_enabled = os.getenv("CORS_ENABLED", "false").lower() == "true"

    # cors e.g. for development of separate frontend
    if cors_enabled:
        logger.warning(typer.style("🌐 CORS enabled", fg=typer.colors.YELLOW))
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    server_config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(server_config)
    return server_config, server
