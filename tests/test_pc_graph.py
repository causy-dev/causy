import csv
import unittest

from causy.graph import PCGraph
from causy.cli import show_edges


class PCTestTestCase(unittest.TestCase):
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
