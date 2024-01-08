import torch
import torch.nn as nn

# The number of characters in the input language.
NUM_CHARS = 27
INPUT_LEN = 4

# CHANNELS_1 = 100
# IN_WIDTH_1 = 5

# CHANNELS_2 = 100
# IN_WIDTH_2 = 5

CHANNELS_1 = 50
CHANNELS_2 = 50
OUT_CHANNELS = NUM_CHARS


# A simple fully-connected model for predicting the next letter.
class SimpleLetterModel(nn.Module):
    def __init__(self):
        super(SimpleLetterModel, self).__init__()

        self.layer_1 = nn.Linear(
            in_features=INPUT_LEN * NUM_CHARS, out_features=CHANNELS_1
        )
        self.act_1 = nn.ReLU()

        self.layer_2 = nn.Linear(in_features=CHANNELS_1, out_features=CHANNELS_2)
        self.act_2 = nn.ReLU()

        self.output = nn.Linear(in_features=CHANNELS_2, out_features=OUT_CHANNELS)

    def forward(self, x):
        # print(f"x.shape: {x.shape}")
        N = x.shape[0]
        x = x.reshape(N, -1)
        # print(f"x.shape: {x.shape}")

        x = self.layer_1(x)
        # print(f"x.shape: {x.shape}")
        x = self.act_1(x)

        x = self.layer_2(x)
        # print(f"x.shape: {x.shape}")
        x = self.act_2(x)

        return self.output(x)
