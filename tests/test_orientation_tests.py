from causy.causal_discovery.constraint.algorithms.pc import PC_ORIENTATION_RULES
from causy.common_pipeline_steps.exit_conditions import ExitOnNoActions
from causy.common_pipeline_steps.logic import Loop
from causy.edge_types import DirectedEdge, UndirectedEdge
from causy.graph import GraphManager
from causy.models import TestResultAction, TestResult, Algorithm
from causy.causal_discovery.constraint.orientation_rules.pc import (
    ColliderTest,
    NonColliderTest,
    FurtherOrientTripleTest,
    OrientQuadrupleTest,
    FurtherOrientQuadrupleTest,
    ColliderTestConflictResolutionStrategies,
)
from causy.graph_model import graph_model_factory

from tests.utils import CausyTestCase


class OrientationRuleTestCase(CausyTestCase):
    def test_collider_test(self):
        pipeline = [ColliderTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[DirectedEdge, UndirectedEdge],
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
        self.assertFalse(model.graph.edge_exists(x, z))

    def test_collider_test_2(self):
        pipeline = [ColliderTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[DirectedEdge(), UndirectedEdge()],
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
                data={"separatedBy": [y]},
            ),
        )
        model.execute_pipeline_steps()
        self.assertFalse(model.graph.only_directed_edge_exists(x, y))
        self.assertFalse(model.graph.only_directed_edge_exists(z, y))
        self.assertTrue(model.graph.edge_exists(x, y))
        self.assertTrue(model.graph.edge_exists(z, y))
        self.assertFalse(model.graph.edge_exists(x, z))

    def test_collider_test_3(self):
        pipeline = [ColliderTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[DirectedEdge(), UndirectedEdge()],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [0, 1, 2])
        y = model.graph.add_node("Y", [3, 4, 5])
        z = model.graph.add_node("Z", [6, 7, 8])
        a = model.graph.add_node("A", [9, 10, 11])
        model.graph.add_edge(x, y, {})
        model.graph.add_edge(z, y, {})
        model.graph.add_edge(x, a, {})
        model.graph.add_edge(z, a, {})
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
        model.graph.add_edge_history(
            y,
            a,
            TestResult(
                u=y,
                v=a,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={"separatedBy": [x, z]},
            ),
        )
        model.execute_pipeline_steps()
        self.assertTrue(model.graph.only_directed_edge_exists(x, y))
        self.assertTrue(model.graph.only_directed_edge_exists(z, y))
        self.assertTrue(model.graph.only_directed_edge_exists(x, a))
        self.assertTrue(model.graph.only_directed_edge_exists(z, a))
        self.assertFalse(model.graph.edge_exists(x, z))

    def test_collider_test_4(self):
        pipeline = [ColliderTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[DirectedEdge(), UndirectedEdge()],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [0, 1, 2])
        y = model.graph.add_node("Y", [3, 4, 5])
        z = model.graph.add_node("Z", [6, 7, 8])
        a = model.graph.add_node("A", [9, 10, 11])
        model.graph.add_edge(x, y, {})
        model.graph.add_edge(z, y, {})
        model.graph.add_edge(x, a, {})
        model.graph.add_edge(z, a, {})
        model.graph.add_edge_history(
            x,
            z,
            TestResult(
                u=x,
                v=z,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={"separatedBy": [a]},
            ),
        )
        model.graph.add_edge_history(
            y,
            a,
            TestResult(
                u=y,
                v=a,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={"separatedBy": [x, z]},
            ),
        )
        model.execute_pipeline_steps()
        self.assertTrue(model.graph.only_directed_edge_exists(x, y))
        self.assertTrue(model.graph.only_directed_edge_exists(z, y))
        self.assertTrue(model.graph.edge_exists(x, y))
        self.assertTrue(model.graph.edge_exists(z, y))
        self.assertFalse(model.graph.only_directed_edge_exists(x, a))
        self.assertFalse(model.graph.only_directed_edge_exists(z, a))
        self.assertTrue(model.graph.edge_exists(x, a))
        self.assertTrue(model.graph.edge_exists(z, a))
        self.assertFalse(model.graph.edge_exists(x, z))

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
                edge_types=[DirectedEdge(), UndirectedEdge()],
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
                edge_types=[DirectedEdge(), UndirectedEdge()],
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
                edge_types=[DirectedEdge(), UndirectedEdge()],
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
            z,
            TestResult(
                u=x,
                v=z,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={"separatedBy": [y]},
            ),
        )
        model.execute_pipeline_steps()
        self.assertFalse(model.graph.only_directed_edge_exists(x, y))
        self.assertFalse(model.graph.only_directed_edge_exists(z, y))
        self.assertTrue(model.graph.undirected_edge_exists(x, y))
        self.assertTrue(model.graph.undirected_edge_exists(y, z))

    def test_collider_multiple_colliders(self):
        pipeline = [ColliderTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[DirectedEdge(), UndirectedEdge()],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [])
        y = model.graph.add_node("Y", [])
        z = model.graph.add_node("Z", [])
        a = model.graph.add_node("A", [])
        model.graph.add_edge(x, y, {})
        model.graph.add_edge(z, y, {})
        model.graph.add_edge(x, a, {})
        model.graph.add_edge(z, a, {})
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
        self.assertTrue(model.graph.only_directed_edge_exists(x, a))
        self.assertTrue(model.graph.only_directed_edge_exists(z, a))
        self.assertTrue(model.graph.only_directed_edge_exists(x, y))
        self.assertTrue(model.graph.only_directed_edge_exists(z, y))

    def test_collider_prioritize_collider_rules(self):
        pipeline = [
            ColliderTest(
                conflict_resolution_strategy=ColliderTestConflictResolutionStrategies.KEEP_LAST
            )
        ]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[DirectedEdge(), UndirectedEdge()],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [])
        y = model.graph.add_node("Y", [])
        z = model.graph.add_node("Z", [])
        a = model.graph.add_node("A", [])
        model.graph.add_edge(x, y, {})
        model.graph.add_edge(z, y, {})
        model.graph.add_edge(x, a, {})
        model.graph.add_edge(z, a, {})
        model.graph.remove_directed_edge(a, x)
        model.graph.remove_directed_edge(a, z)
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
        model.graph.add_edge_history(
            a,
            y,
            TestResult(
                u=x,
                v=z,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={"separatedBy": []},
            ),
        )
        model.execute_pipeline_steps()
        self.assertTrue(model.graph.edge_exists(y, z))
        self.assertTrue(model.graph.edge_exists(a, x))
        self.assertTrue(model.graph.edge_exists(a, z))
        self.assertTrue(model.graph.edge_exists(x, y))

    def test_collider_prioritize_collider_rules_2(self):
        pipeline = [
            ColliderTest(
                conflict_resolution_strategy=ColliderTestConflictResolutionStrategies.KEEP_FIRST
            )
        ]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[DirectedEdge(), UndirectedEdge()],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        x = model.graph.add_node("X", [])
        y = model.graph.add_node("Y", [])
        z = model.graph.add_node("Z", [])
        a = model.graph.add_node("A", [])
        model.graph.add_edge(x, y, {})
        model.graph.add_edge(z, y, {})
        model.graph.add_edge(x, a, {})
        model.graph.add_edge(z, a, {})
        model.graph.remove_directed_edge(a, x)
        model.graph.remove_directed_edge(a, z)
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
        model.graph.add_edge_history(
            a,
            y,
            TestResult(
                u=x,
                v=z,
                action=TestResultAction.REMOVE_EDGE_UNDIRECTED,
                data={"separatedBy": []},
            ),
        )
        model.execute_pipeline_steps()
        self.assertTrue(model.graph.edge_exists(y, z))
        self.assertTrue(model.graph.edge_exists(a, x))
        self.assertTrue(model.graph.edge_exists(a, z))
        self.assertTrue(model.graph.edge_exists(x, y))

    def test_non_collider_test(self):
        pipeline = [NonColliderTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[DirectedEdge(), UndirectedEdge()],
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

    def test_non_collider_test_auto_mpg_graph(self):
        pipeline = [NonColliderTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[DirectedEdge(), UndirectedEdge()],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        acceleration = model.graph.add_node("acceleration", [])
        horsepower = model.graph.add_node("horsepower", [])
        mpg = model.graph.add_node("mpg", [])
        cylinders = model.graph.add_node("cylinders", [])
        displacement = model.graph.add_node("displacement", [])
        weight = model.graph.add_node("weight", [])
        model.graph.add_edge(mpg, weight, {})
        model.graph.add_edge(weight, displacement, {})
        model.graph.add_edge(displacement, cylinders, {})

        model.graph.add_directed_edge(acceleration, horsepower, {})
        model.graph.add_directed_edge(horsepower, displacement, {})
        model.graph.add_directed_edge(mpg, horsepower, {})

        model.execute_pipeline_steps()
        self.assertTrue(model.graph.edge_of_type_exists(displacement, cylinders, DirectedEdge()))
        self.assertTrue(model.graph.edge_of_type_exists(displacement, weight, DirectedEdge()))

    def test_non_collider_loop_auto_mpg_graph(self):
        pipeline = [*PC_ORIENTATION_RULES]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[DirectedEdge(), UndirectedEdge()],
                name="TestCollider",
            )
        )()
        model.graph = GraphManager()
        acceleration = model.graph.add_node("acceleration", [])
        horsepower = model.graph.add_node("horsepower", [])
        mpg = model.graph.add_node("mpg", [])
        cylinders = model.graph.add_node("cylinders", [])
        displacement = model.graph.add_node("displacement", [])
        weight = model.graph.add_node("weight", [])
        model.graph.add_edge(mpg, weight, {})
        model.graph.add_edge(weight, displacement, {})
        model.graph.add_edge(displacement, cylinders, {})
        model.graph.add_edge(horsepower, displacement, {})

        model.graph.add_directed_edge(acceleration, horsepower, {})
        model.graph.add_directed_edge(mpg, horsepower, {})

        model.execute_pipeline_steps()
        self.assertTrue(model.graph.edge_of_type_exists(displacement, cylinders, DirectedEdge()))
        self.assertTrue(model.graph.edge_of_type_exists(displacement, weight, DirectedEdge()))



    def test_further_orient_triple_test(self):
        pipeline = [FurtherOrientTripleTest()]
        model = graph_model_factory(
            Algorithm(
                pipeline_steps=pipeline,
                edge_types=[DirectedEdge(), UndirectedEdge()],
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
                edge_types=[DirectedEdge(), UndirectedEdge()],
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
                edge_types=[DirectedEdge(), UndirectedEdge()],
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
