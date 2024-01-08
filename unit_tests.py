import unittest

import markov_chain


class MarkovChainTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_add_prefix_info_1(self):
        p1 = {"AB": {"A": 1, "S": 4}, "BAG": {"B": 3, "T": 2}}
        p2 = {"AB": {"A": 5, "Q": 10}, "HUB": {"T": 1}}
        markov_chain.add_prefix_info(p1=p1, p2=p2)
        exp_p1 = {
            "AB": {"A": 6, "S": 4, "Q": 10},
            "BAG": {"B": 3, "T": 2},
            "HUB": {"T": 1},
        }
        self.assertDictEqual(p1, exp_p1)
    
if __name__ == "__main__":
    unittest.main()