from causy.graph import GraphManager
from causy.graph_model import graph_model_factory
from causy.interfaces import TestResult, TestResultAction, CausyAlgorithm
from causy.orientation_rules.fci import ColliderRuleFCI

from tests.utils import CausyTestCase


class OrientationTestCase(CausyTestCase):
    def test_collider_rule_fci(self):
        pipeline = [ColliderRuleFCI()]
        model = graph_model_factory(
            CausyAlgorithm(
                pipeline_steps=pipeline,
                edge_types=[],
                name="FCIColliderRule",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [0, 1, 2])
        y = model.graph.add_node("Y", [3, 4, 5])
        z = model.graph.add_node("Z", [6, 7, 8])
        model.graph.add_edge(x, y, {"edge_type": "either directed or undirected"})
        model.graph.add_edge(z, y, {"edge_type": "either directed or undirected"})
        model.graph.add_edge_history(
            x,
            y,
            TestResult(
                u=x,
                v=z,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={"separatedBy": []},
            ),
        )
        model.execute_pipeline_steps()
        self.assertEqual(model.graph.edge_value(x, y), {"edge_type": None})
        self.assertEqual(model.graph.edge_value(z, y), {"edge_type": None})
        self.assertEqual(
            model.graph.edge_value(y, x), {"edge_type": "either directed or undirected"}
        )
        self.assertEqual(
            model.graph.edge_value(y, z), {"edge_type": "either directed or undirected"}
        )
