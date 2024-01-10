import torch
import torch.nn as nn

# The number of characters in the input language.
NUM_CHARS = 27
INPUT_LEN = 10

IN_CH_1 = NUM_CHARS
OUT_CH_1 = 100

IN_CH_2 = IN_CH_1 + OUT_CH_1
OUT_CH_2 = 100

IN_CH_3 = IN_CH_2 + OUT_CH_2
OUT_CH_3 = 100

# Use a fully-connected layer for the output, with no pooling.
IN_CH_OUT = INPUT_LEN * (IN_CH_3 + OUT_CH_3)
OUT_CH_OUT = NUM_CHARS


# A convolutional model for predicting the next letter.
class ConvLetterModel(nn.Module):
    def __init__(self):
        super(ConvLetterModel, self).__init__()

        self.conv_1 = nn.Conv1d(
            in_channels=IN_CH_1, out_channels=OUT_CH_1, kernel_size=5, padding=2
        )
        self.act_1 = nn.ReLU()

        self.conv_2 = nn.Conv1d(
            in_channels=IN_CH_2, out_channels=OUT_CH_2, kernel_size=5, padding=2
        )
        self.act_2 = nn.ReLU()

        self.conv_3 = nn.Conv1d(
            in_channels=IN_CH_3, out_channels=OUT_CH_3, kernel_size=5, padding=2
        )
        self.act_3 = nn.ReLU()

        self.out = nn.Linear(in_features=IN_CH_OUT, out_features=OUT_CH_OUT)

    def forward(self, x):
        # Do the convolution layers.
        c1 = self.conv_1(x)
        c1 = self.act_1(c1)
        x = torch.concat([x, c1], dim=1)

        c2 = self.conv_2(x)
        c2 = self.act_2(c2)
        x = torch.concat([x, c2], dim=1)

        c3 = self.conv_3(x)
        c3 = self.act_3(c3)
        x = torch.concat([x, c3], dim=1)

        # Flatten the input for the final layer.
        N = x.shape[0]
        x = x.reshape(N, -1)

        out = self.out(x)
        return out
