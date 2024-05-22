from fastapi.testclient import TestClient

from causy.data_loader import DataLoaderReference, DataLoaderType
from causy.models import AlgorithmReference, AlgorithmReferenceType
from causy.ui.server import _create_ui_app, _set_workspace, _set_model
from causy.workspaces.models import Workspace, Experiment
from tests.utils import CausyTestCase


class UIApiTestCase(CausyTestCase):
    def test_status_endpoint(self):
        _set_workspace(None)
        _set_model(None)
        app = _create_ui_app(with_static=False)
        client = TestClient(app)
        response = client.get("/api/v1/status")
        result = response.json()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(response.status_code, 200)

        self.assertEqual(result["causy_version"], "0.1.0")
        self.assertEqual(result["model_loaded"], False)
        self.assertEqual(result["workspace_loaded"], False)

        _set_workspace(
            Workspace(
                name="test_workspace",
                author="test_author",
                pipelines=None,
                experiments=None,
                data_loaders=None,
            )
        )

        response = client.get("/api/v1/status")
        result = response.json()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(response.status_code, 200)

        self.assertEqual(result["model_loaded"], False)
        self.assertEqual(result["workspace_loaded"], True)

    def test_workspace(self):
        app = _create_ui_app(with_static=False)
        client = TestClient(app)
        _set_workspace(None)
        response = client.get("/api/v1/workspace")
        self.assertEqual(response.status_code, 404)

        _set_workspace(
            Workspace(
                name="test_workspace",
                author="test_author",
                pipelines={
                    "PC": AlgorithmReference(
                        type=AlgorithmReferenceType.NAME, reference="PC"
                    )
                },
                experiments={
                    "test_experiment": Experiment(
                        pipeline="PC", data_loader="data_loader", variables=None
                    )
                },
                data_loaders={
                    "test_data_loader": DataLoaderReference(
                        type=DataLoaderType.JSON, reference="data_loader"
                    )
                },
            )
        )

        response = client.get("/api/v1/workspace")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["name"], "test_workspace")
        self.assertEqual(result["author"], "test_author")
        self.assertEqual(
            result["pipelines"], {"PC": {"type": "name", "reference": "PC"}}
        )

        self.assertEqual(
            result["data_loaders"],
            {
                "test_data_loader": {
                    "type": "json",
                    "reference": "data_loader",
                    "options": None,
                }
            },
        )

        self.assertEqual(
            result["experiments"],
            {
                "test_experiment": {
                    "pipeline": "PC",
                    "data_loader": "data_loader",
                    "variables": None,
                }
            },
        )

    def test_get_model(self):
        app = _create_ui_app(with_static=False)
        client = TestClient(app)
        result = {
            "algorithm": {"type": "name", "reference": "PC"},
            "nodes": {},
            "edges": [],
            "action_history": [],
            "data_loader": {
                "type": "json",
                "reference": "data_loader",
                "options": None,
            },
            "variables": {"test": "test"},
            "result": {"test": "test"},
        }
        _set_model(result)
        response = client.get("/api/v1/model")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["algorithm"], {"type": "name", "reference": "PC"})

    def test_get_algorithm(self):
        app = _create_ui_app(with_static=False)
        client = TestClient(app)

        response = client.get("/api/v1/algorithm/name/PC")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["name"], "PC")

    def test_get_algorithm_invalid(self):
        app = _create_ui_app(with_static=False)
        client = TestClient(app)

        response = client.get("/api/v1/algorithm/name/INVALID")
        self.assertEqual(response.status_code, 400)

        response = client.get("/api/v1/algorithm/python_module/INVALID")
        self.assertEqual(response.status_code, 400)

        response = client.get("/api/v1/algorithm/name/..PC")
        self.assertEqual(response.status_code, 400)

    def test_get_experiments(self):
        app = _create_ui_app(with_static=False)
        client = TestClient(app)
        _set_workspace(None)

        response = client.get("/api/v1/experiments")
        self.assertEqual(response.status_code, 404)

        _set_workspace(
            Workspace(
                name="test_workspace",
                author="test_author",
                pipelines={
                    "PC": AlgorithmReference(
                        type=AlgorithmReferenceType.NAME, reference="PC"
                    )
                },
                experiments={
                    "test_experiment": Experiment(
                        pipeline="PC", data_loader="data_loader", variables=None
                    )
                },
                data_loaders={
                    "test_data_loader": DataLoaderReference(
                        type=DataLoaderType.JSON, reference="data_loader"
                    )
                },
            )
        )

        response = client.get("/api/v1/experiments")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "test_experiment")
        self.assertEqual(result[0]["pipeline"], "PC")
        self.assertEqual(result[0]["data_loader"], "data_loader")
        self.assertEqual(result[0]["variables"], None)
        self.assertEqual(len(result[0]["versions"]), 0)

    def test_get_latest_experiment(self):
        app = _create_ui_app(with_static=False)
        client = TestClient(app)
        _set_workspace(None)

        response = client.get("/api/v1/experiments/test_experiment/latest")
        self.assertEqual(response.status_code, 404)

        _set_workspace(
            Workspace(
                name="test_workspace",
                author="test_author",
                pipelines={
                    "PC": AlgorithmReference(
                        type=AlgorithmReferenceType.NAME, reference="PC"
                    )
                },
                experiments={
                    "test_experiment": Experiment(
                        pipeline="PC", data_loader="data_loader", variables=None
                    )
                },
                data_loaders={
                    "test_data_loader": DataLoaderReference(
                        type=DataLoaderType.JSON, reference="data_loader"
                    )
                },
            )
        )

        response = client.get("/api/v1/experiments/test_experiment/latest")
        self.assertEqual(response.status_code, 400)
        result = response.json()
        self.assertEqual(
            result["detail"], "Experiment test_experiment not found in the file system"
        )

        # TODO: add test for when the experiment is found (hmw to mock the file system?)

    def test_get_experiment(self):
        app = _create_ui_app(with_static=False)
        client = TestClient(app)
        _set_workspace(None)

        response = client.get("/api/v1/experiments/test_experiment/1")
        self.assertEqual(response.status_code, 404)

        _set_workspace(
            Workspace(
                name="test_workspace",
                author="test_author",
                pipelines={
                    "PC": AlgorithmReference(
                        type=AlgorithmReferenceType.NAME, reference="PC"
                    )
                },
                experiments={
                    "test_experiment": Experiment(
                        pipeline="PC", data_loader="data_loader", variables=None
                    )
                },
                data_loaders={
                    "test_data_loader": DataLoaderReference(
                        type=DataLoaderType.JSON, reference="data_loader"
                    )
                },
            )
        )

        response = client.get("/api/v1/experiments/test_experiment/1")
        self.assertEqual(response.status_code, 400)
        result = response.json()
        self.assertEqual(
            result["detail"], "Version 1 not found for experiment test_experiment"
        )

        # TODO: add test for when the experiment is found (hmw to mock the file system?)
