import os
from datetime import datetime
from importlib.metadata import version

import fastapi
import typer
import uvicorn
from typing import Any, Dict, Optional, Union, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, UUID4
from starlette.staticfiles import StaticFiles

import logging

from causy.interfaces import (
    NodeInterface,
)
from causy.serialization import load_algorithm_by_reference, load_json
from causy.workspaces.cli import (
    _current_workspace,
    _load_latest_experiment_result,
    _load_experiment_result,
    _load_experiment_versions,
)
from causy.workspaces.models import Workspace, Experiment
from causy.models import (
    CausyResult,
    CausyAlgorithmReference,
    CausyAlgorithmReferenceType,
    CausyAlgorithm,
)

logger = logging.getLogger(__name__)

API_ROUTES = APIRouter()

MODEL = None


class NodePosition(BaseModel):
    x: Optional[float]
    y: Optional[float]


class ExperimentVersion(BaseModel):
    version: int
    name: str


class ExtendedExperiment(Experiment):
    versions: Optional[List[ExperimentVersion]] = None
    name: str = None


class PositionedNode(NodeInterface):
    position: Optional[NodePosition] = None


class CausyExtendedResult(CausyResult):
    nodes: Dict[Union[UUID4, str], PositionedNode]
    version: Optional[int] = None


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


@API_ROUTES.get("/model", response_model=CausyExtendedResult)
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


@API_ROUTES.get(
    "/experiments/{experiment_name}/latest", response_model=CausyExtendedResult
)
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

    experiment["algorithm"] = CausyAlgorithmReference(**experiment["algorithm"])
    experiment["version"] = version
    experiment = CausyExtendedResult(**experiment)

    return experiment


@API_ROUTES.get(
    "/experiments/{experiment_name}/{version_number}",
    response_model=CausyExtendedResult,
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

    experiment["algorithm"] = CausyAlgorithmReference(**experiment["algorithm"])
    experiment["version"] = version_number
    experiment = CausyExtendedResult(**experiment)

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


@API_ROUTES.get(
    "/algorithm/{reference_type}/{reference}", response_model=CausyAlgorithm
)
async def get_algorithm(reference_type: str, reference: str):
    """Get the current algorithm."""
    if reference.startswith("/") or ".." in reference:
        raise HTTPException(400, "Invalid reference")

    if reference_type not in CausyAlgorithmReferenceType.__members__.values():
        raise HTTPException(400, "Invalid reference type")

    try:
        algorithm = load_algorithm_by_reference(reference_type, reference)
        return algorithm
    except Exception as e:
        raise HTTPException(400, str(e))


def server(result: Dict[str, Any] = None, workspace=None):
    """Create the FastAPI server."""
    app = fastapi.FastAPI(
        title="causy-api",
        version=version("causy"),
        description="causys internal api to serve data from the result files",
    )
    global MODEL
    global WORKSPACE
    if result:
        result["algorithm"] = CausyAlgorithmReference(**result["algorithm"])
        MODEL = CausyExtendedResult(**result)

    if workspace:
        WORKSPACE = workspace

    if not MODEL and not WORKSPACE:
        raise ValueError("No model or workspace provided")

    app.include_router(API_ROUTES, prefix="/api/v1", tags=["api"])
    app.mount(
        "",
        StaticFiles(
            directory=os.path.join(os.path.dirname(__file__), "static"), html=True
        ),
        name="static",
    )

    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", "8000"))
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
