import torch

from causy.causal_discovery.constraint.algorithms import PC
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference

from tests.utils import CausyTestCase


class GraphModelTestCase(CausyTestCase):
    def test_initialize_graph_model_with_list(self):
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
            ]
        )

        data, graph = model.generate(100)

        pc = PC()
        pc.create_graph_from_data(data)
        self.assertEqual(pc.graph.nodes["X"].values.shape, torch.Size([100]))

    def test_initialize_graph_model_with_dict(self):
        model = IIDSampleGenerator(
            edges=[
                SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
                SampleEdge(NodeReference("Y"), NodeReference("Z"), 7),
            ]
        )

        data = model._generate_shaped_data(100)

        pc = PC()
        pc.create_graph_from_data(data)
        self.assertEqual(pc.graph.nodes["X"].values.shape, torch.Size([100]))
