import unittest

import markov_chain
import huffman_coding


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

class HuffmanCodingTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_huffman_tree(self):
        tok_to_count = {"A": 100, "B": 50, "C": 30, "D": 25}
        tree = huffman_coding.get_huffman_tree(tok_to_count=tok_to_count)
        msg = "ADBCBADCB"
        bits = tree.encode(msg)
        self.assertIsNotNone(bits)
        assert bits is not None
        decoded = tree.decode(bits)
        self.assertEqual(decoded, msg)
    
if __name__ == "__main__":
    unittest.main()