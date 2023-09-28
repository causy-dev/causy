import csv
import unittest
import random

from graph import PCGraph
from utils import sum_lists


def show_edges(graph):
    for u in graph.edges:
        for v in graph.edges[u]:
            print(f"{u.name} -> {v.name}: {graph.edges[u][v]}")


class PCTestTestCase(unittest.TestCase):
    def test_full_graph(self):
        test_data = []

        n = 1000
        sample_size = 5

        samples = {}

        for i in range(sample_size):
            x = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
            noise_y = [random.normalvariate(mu=0.0, sigma=1.0) for _ in range(n)]
            samples[str(i)] = sum_lists([5 * x_val for x_val in x], noise_y)

        for i in range(n):
            entry = {}
            for key in samples.keys():
                entry[key] = samples[key][i]
            test_data.append(entry)

        tst = PCGraph()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        show_edges(tst.graph)

    def test_with_rki_data(self):
        with open("./tests/fixtures/rki-data.csv") as f:
            data = csv.DictReader(f)
            test_data = []
            for row in data:
                for k in row.keys():
                    if row[k] == "":
                        row[k] = 0.0
                    row[k] = float(row[k])

                test_data.append(row)

        tst = PCGraph()
        tst.create_graph_from_data(test_data)
        tst.create_all_possible_edges()
        tst.execute_pipeline_steps()
        show_edges(tst.graph)


if __name__ == "__main__":
    unittest.main()
