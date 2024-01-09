import torch
import torch.nn as nn

# The number of characters in the input language.
NUM_CHARS = 27
INPUT_LEN = 10

# CHANNELS_1 = 100
# IN_WIDTH_1 = 5

# CHANNELS_2 = 100
# IN_WIDTH_2 = 5

IN_1 = INPUT_LEN * NUM_CHARS
CHANNELS_1 = 100

IN_2 = IN_1 + CHANNELS_1
CHANNELS_2 = 100

IN_3 = IN_2 + CHANNELS_2
CHANNELS_3 = 100

IN_OUT = IN_3 + CHANNELS_3
OUT_CHANNELS = NUM_CHARS


# A simple fully-connected model for predicting the next letter.
class SimpleLetterModel(nn.Module):
    def __init__(self):
        super(SimpleLetterModel, self).__init__()

        self.layer_1 = nn.Linear(
            in_features=IN_1, out_features=CHANNELS_1
        )
        self.act_1 = nn.ReLU()

        self.layer_2 = nn.Linear(in_features=IN_2, out_features=CHANNELS_2)
        self.act_2 = nn.ReLU()

        self.layer_3 = nn.Linear(in_features=IN_3, out_features=CHANNELS_3)
        self.act_3 = nn.ReLU()

        self.output = nn.Linear(in_features=IN_OUT, out_features=OUT_CHANNELS)

    def forward(self, x):
        N = x.shape[0]
        x = x.reshape(N, -1)

        l1 = self.layer_1(x)
        l1 = self.act_1(l1)
        # print(f"x.shape: {x.shape}")
        x = torch.concat([x, l1], dim=1)
        # print(f"x.shape: {x.shape}")

        l2 = self.layer_2(x)
        l2 = self.act_2(l2)
        x = torch.concat([x, l2], dim=1)
        # print(f"x.shape: {x.shape}")

        l3 = self.layer_3(x)
        l3 = self.act_3(l3)
        x = torch.concat([x, l3], dim=1)
        # print(f"x.shape: {x.shape}")

        return self.output(x)
