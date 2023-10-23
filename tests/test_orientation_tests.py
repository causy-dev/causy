import unittest

from causy.graph import graph_model_factory, UndirectedGraph
from causy.orientation_tests import ColliderTest


class OrientationRuleTestCase(unittest.TestCase):
    def test_collider_test(self):
        pipeline = [ColliderTest()]
        model = graph_model_factory(pipeline_steps=pipeline)()
        model.graph = UndirectedGraph()
        x = model.graph.add_node("X", [0,1,2])
        y = model.graph.add_node("Y", [3, 4, 5])
        z = model.graph.add_node("Z", [6, 7, 8])
        model.graph.add_edge(x, y,{})
        model.graph.add_edge(z, y,{})
        model.execute_pipeline_steps()
        self.assertTrue(model.graph.only_directed_edge_exists(x, y))
        self.assertTrue(model.graph.only_directed_edge_exists(z, y))

if __name__ == "__main__":
    unittest.main()