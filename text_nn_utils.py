import torch

from consts import *

# Return a one-hot vector encoding the given character. Invalid characters are all zeros.
def char_to_onehot(c: str) -> torch.Tensor:
    index = OUT_CHARS.find(c)
    result = torch.zeros(size=[27])
    if 0 <= index <= 26:
        result[index] = 1
    return result


# Return an array encoding the given string.
def snippet_to_array(snippet: str) -> torch.Tensor:
    onehots = [char_to_onehot(c) for c in snippet]
    return torch.stack(onehots, dim=1)

