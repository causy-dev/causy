import os

import fastapi
import typer
import uvicorn
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter
from pydantic import BaseModel, UUID4
from starlette.staticfiles import StaticFiles

import logging

from causy.interfaces import (
    NodeInterface,
    CausyAlgorithm,
    CausyAlgorithmReference,
    CausyResult,
)
from causy.serialization import load_algorithm_by_reference

logger = logging.getLogger(__name__)

API_ROUTES = APIRouter()

MODEL = None


class NodePosition(BaseModel):
    x: Optional[float]
    y: Optional[float]


class PositionedNode(NodeInterface):
    position: Optional[NodePosition] = None


class CausyExtendedResult(CausyResult):
    nodes: Dict[Union[UUID4, str], PositionedNode]


@API_ROUTES.get("/status", response_model=Dict[str, Any])
async def get_status():
    """Get the current status of the API."""
    return {"status": "ok"}


@API_ROUTES.get("/model", response_model=CausyExtendedResult)
async def get_model():
    """Get the current model."""
    return MODEL


@API_ROUTES.get(
    "/algorithm/{reference_type}/{reference}", response_model=CausyAlgorithm
)
async def get_algorithm(reference_type: str, reference: str):
    """Get the current algorithm."""
    return load_algorithm_by_reference(reference_type, reference)


def server(result: Dict[str, Any]):
    """Create the FastAPI server."""
    app = fastapi.FastAPI(
        title="causy-api",
        version="0.0.1",
        description="causys internal api to serve data from the result files",
    )
    global MODEL
    result["algorithm"] = CausyAlgorithmReference(**result["algorithm"])
    MODEL = CausyExtendedResult(**result)

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
