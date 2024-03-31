import os
from datetime import datetime

import fastapi
import uvicorn
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Json, UUID4, Field
from starlette.staticfiles import StaticFiles

import logging

logger = logging.getLogger(__name__)

API_ROUTES = APIRouter()

MODEL = None


class CausyAlgorithm(BaseModel):
    type: str
    reference: str


class CausyNodePosition(BaseModel):
    x: Optional[float]
    y: Optional[float]


class CausyNode(BaseModel):
    id: UUID4
    name: str
    position: Optional[CausyNodePosition] = None


class CausyEdgeValue(BaseModel):
    metadata: Dict[str, Any] = None
    edge_type: str = None


class CausyEdge(BaseModel):
    class Config:
        allow_population_by_field_name = True
        fields = {"from_field": "from"}

    from_field: CausyNode = Field(alias="from")
    to: CausyNode
    value: CausyEdgeValue


class CausyModel(BaseModel):
    name: str
    created_at: datetime
    algorithm: CausyAlgorithm
    steps: List[Dict[str, Any]]
    nodes: Dict[UUID4, CausyNode]
    edges: List[CausyEdge]


@API_ROUTES.get("/status", response_model=Dict[str, Any])
async def get_status():
    """Get the current status of the API."""
    return {"status": "ok"}


@API_ROUTES.get("/model", response_model=CausyModel)
async def get_model():
    """Get the current model."""
    return MODEL


def server(result: Dict[str, Any]):
    """Create the FastAPI server."""
    app = fastapi.FastAPI(
        title="causy-api",
        version="0.0.1",
        description="causys internal api to serve data from the result files",
    )
    global MODEL
    MODEL = CausyModel(**result)

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
        logger.warning("🌐 CORS enabled")
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
