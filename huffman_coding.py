from typing import cast, Mapping, Generator
from dataclasses import dataclass
from frozendict import frozendict

from bitlist import bitlist


# A branch in the tree.
@dataclass
class HuffmanBranch:
    freq: int
    left: "HuffmanNode"
    right: "HuffmanNode"


# A leaf in the tree.
@dataclass
class HuffmanLeaf:
    freq: int
    tok: str


# Any node in the tree, either a branch or a leaf.
HuffmanNode = HuffmanBranch | HuffmanLeaf


# Return the token-to-bits mapping encoded in the given node.
def get_tok_to_bits(
    node: HuffmanNode, prefix: bitlist | None = None
) -> Generator[tuple[str, bitlist], None, None]:
    if prefix is None:
        prefix = bitlist(length=0)

    if isinstance(node, HuffmanLeaf):
        yield (node.tok, prefix)
    elif isinstance(node, HuffmanBranch):
        yield from get_tok_to_bits(node=node.left, prefix=prefix + bitlist("0"))
        yield from get_tok_to_bits(node=node.right, prefix=prefix + bitlist("1"))


# A Huffman tree.
class HuffmanTree:
    def __init__(self, root: HuffmanBranch):
        self.root = root

        # Get the mapping from tokens to bitlists.
        tok_to_bits = dict[str, bitlist]()
        for tok, bits in get_tok_to_bits(node=self.root):
            tok_to_bits[tok] = bits
        self.tok_to_bits = frozendict(tok_to_bits)

    # Get the bits for the given token.
    def get_bits(self, tok: str) -> bitlist | None:
        return self.tok_to_bits.get(tok, None)

    # Get the token and remaining bits from the given bits.
    def get_tok(self, bits: bitlist) -> tuple[str, bitlist] | None:
        current_node = self.root
        i = 0
        while not isinstance(current_node, HuffmanLeaf):
            try:
                bit = bits[i]
            except IndexError:
                return None
            i += 1
            if bit == 0:
                current_node = current_node.left
            else:
                current_node = current_node.right

        if i >= len(bits):
            return current_node.tok, bitlist(length=0)
        return current_node.tok, bits[i:]  # type: ignore

    # Encode the given message.
    def encode(self, toks: str) -> bitlist | None:
        result = bitlist(length=0)
        for tok in toks:
            bits = self.get_bits(tok=tok)
            if bits is None:
                return None
            result += bits
        return result

    # Decode the given list of bits.
    def decode(self, bits: bitlist) -> str | None:
        result = list[str]()
        bits = bitlist(bits)
        while len(bits) > 0:
            r = self.get_tok(bits=bits)
            if r is None:
                return None
            tok, bits = r
            result.append(tok)
        return "".join(result)


# Get a huffman tree for the given frequencies.
def get_huffman_tree(tok_to_count: Mapping[str, int]) -> HuffmanTree:
    if len(tok_to_count) <= 1:
        raise ValueError()

    # Make the leaves.
    nodes = cast(
        list[HuffmanNode],
        [HuffmanLeaf(freq=freq, tok=tok) for tok, freq in tok_to_count.items()],
    )
    nodes.sort(key=lambda n: n.freq, reverse=True)

    while len(nodes) > 1:
        # TODO combine bottom two
        nodes, b1, b2 = nodes[:-2], nodes[-2], nodes[-1]
        b_branch = HuffmanBranch(freq=b1.freq + b2.freq, left=b1, right=b2)
        nodes.append(b_branch)
        nodes.sort(key=lambda n: n.freq, reverse=True)

    return HuffmanTree(root=nodes[0])  # type: ignore
