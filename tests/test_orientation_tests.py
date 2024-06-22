from causy.common_pipeline_steps.exit_conditions import ExitOnNoActions
from causy.common_pipeline_steps.logic import Loop
from causy.graph import GraphManager
from causy.models import TestResultAction, TestResult, Algorithm
from causy.causal_discovery.constraint.orientation_rules.pc import (
    ColliderTest,
    NonColliderTest,
    FurtherOrientTripleTest,
    OrientQuadrupleTest,
    FurtherOrientQuadrupleTest,
)
from causy.graph_model import graph_model_factory

from tests.utils import CausyTestCase


class OrientationRuleTestCase(CausyTestCase):
    def test_collider_test(self):
        pipeline = [ColliderTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [0, 1, 2])
        y = model.graph.add_node("Y", [3, 4, 5])
        z = model.graph.add_node("Z", [6, 7, 8])
        model.graph.add_edge(x, y, {})
        model.graph.add_edge(z, y, {})
        model.graph.add_edge_history(
            x,
            z,
            TestResult(
                u=x,
                v=z,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={"separatedBy": []},
            ),
        )
        model.execute_pipeline_steps()
        self.assertTrue(model.graph.only_directed_edge_exists(x, y))
        self.assertTrue(model.graph.only_directed_edge_exists(z, y))

    def test_collider_test_multiple_orientation_rules(self):
        pipeline = [
            ColliderTest(),
            NonColliderTest(),
            FurtherOrientTripleTest(),
            OrientQuadrupleTest(),
            FurtherOrientQuadrupleTest(),
        ]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [0, 1, 2])
        y = model.graph.add_node("Y", [3, 4, 5])
        z = model.graph.add_node("Z", [6, 7, 8])
        model.graph.add_edge(x, y, {})
        model.graph.add_edge(z, y, {})
        model.graph.add_edge_history(
            x,
            z,
            TestResult(
                u=x,
                v=z,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={"separatedBy": []},
            ),
        )
        model.execute_pipeline_steps()
        self.assertTrue(model.graph.only_directed_edge_exists(x, y))
        self.assertTrue(model.graph.only_directed_edge_exists(z, y))

    def test_collider_test_multiple_orientation_rules_loop(self):
        pipeline = [
            ColliderTest(),
            Loop(
                pipeline_steps=[
                    NonColliderTest(),
                    FurtherOrientTripleTest(),
                    OrientQuadrupleTest(),
                    FurtherOrientQuadrupleTest(),
                ],
                exit_condition=ExitOnNoActions(),
            ),
        ]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [0, 1, 2])
        y = model.graph.add_node("Y", [3, 4, 5])
        z = model.graph.add_node("Z", [6, 7, 8])
        model.graph.add_edge(x, y, {})
        model.graph.add_edge(z, y, {})
        model.graph.add_edge_history(
            x,
            z,
            TestResult(
                u=x,
                v=z,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={"separatedBy": []},
            ),
        )
        model.execute_pipeline_steps()
        self.assertTrue(model.graph.only_directed_edge_exists(x, y))
        self.assertFalse(model.graph.directed_edge_exists(y, x))
        self.assertTrue(model.graph.only_directed_edge_exists(z, y))
        self.assertFalse(model.graph.directed_edge_exists(y, z))
        self.assertFalse(model.graph.edge_exists(x, z))

    def test_collider_test_with_nonempty_separation_set(self):
        pipeline = [ColliderTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [])
        y = model.graph.add_node("Y", [])
        z = model.graph.add_node("Z", [])
        model.graph.add_edge(x, y, {})
        model.graph.add_edge(z, y, {})
        model.graph.add_edge_history(
            x,
            y,
            TestResult(
                u=x,
                v=z,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={"separatedBy": [y]},
            ),
        )
        self.assertTrue(model.graph.undirected_edge_exists(x, y))
        self.assertTrue(model.graph.undirected_edge_exists(y, z))

    def test_non_collider_test(self):
        pipeline = [NonColliderTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [])
        y = model.graph.add_node("Y", [])
        z = model.graph.add_node("Z", [])
        model.graph.add_edge(x, y, {})
        model.graph.remove_directed_edge(y, x)
        model.graph.add_edge(z, y, {})
        model.execute_pipeline_steps()
        self.assertTrue(model.graph.only_directed_edge_exists(x, y))
        self.assertTrue(model.graph.only_directed_edge_exists(y, z))

    def test_further_orient_triple_test(self):
        pipeline = [FurtherOrientTripleTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [])
        y = model.graph.add_node("Y", [])
        z = model.graph.add_node("Z", [])
        model.graph.add_edge(x, y, {})
        model.graph.add_edge(x, z, {})
        model.graph.add_edge(z, y, {})
        model.graph.remove_directed_edge(y, x)
        model.graph.remove_directed_edge(z, y)
        self.assertTrue(model.graph.only_directed_edge_exists(x, y))
        self.assertTrue(model.graph.only_directed_edge_exists(y, z))
        self.assertTrue(model.graph.undirected_edge_exists(x, z))
        model.execute_pipeline_steps()
        self.assertTrue(model.graph.only_directed_edge_exists(x, z))

    def test_orient_quadruple_test(self):
        pipeline = [OrientQuadrupleTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [])
        y = model.graph.add_node("Y", [])
        z = model.graph.add_node("Z", [])
        w = model.graph.add_node("W", [])
        model.graph.add_edge(x, y, {})
        model.graph.add_edge(x, w, {})
        model.graph.add_edge(w, z, {})
        model.graph.add_edge(z, y, {})
        model.graph.add_edge(x, z, {})
        model.graph.remove_directed_edge(z, y)
        model.graph.remove_directed_edge(z, w)
        self.assertTrue(model.graph.only_directed_edge_exists(y, z))
        self.assertTrue(model.graph.only_directed_edge_exists(w, z))
        self.assertTrue(model.graph.undirected_edge_exists(x, z))
        model.execute_pipeline_steps()
        self.assertTrue(model.graph.only_directed_edge_exists(x, z))

    def test_further_orient_quadruple_test(self):
        pipeline = [FurtherOrientQuadrupleTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [])
        y = model.graph.add_node("Y", [])
        z = model.graph.add_node("Z", [])
        w = model.graph.add_node("W", [])
        model.graph.add_edge(x, y, {})
        model.graph.add_edge(x, w, {})
        model.graph.add_edge(x, z, {})
        model.graph.add_edge(w, z, {})
        model.graph.add_edge(w, y, {})
        model.graph.remove_directed_edge(w, y)
        model.graph.remove_directed_edge(z, w)
        self.assertTrue(model.graph.only_directed_edge_exists(y, w))
        self.assertTrue(model.graph.only_directed_edge_exists(w, z))
        self.assertTrue(model.graph.undirected_edge_exists(x, z))
        model.execute_pipeline_steps()
        self.assertTrue(model.graph.edge_exists(x, z))
        self.assertFalse(model.graph.undirected_edge_exists(x, z))
        self.assertFalse(model.graph.only_directed_edge_exists(z, x))
        self.assertTrue(model.graph.only_directed_edge_exists(x, z))
