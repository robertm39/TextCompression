import torch
import torch.nn as nn
import torch.nn.init as init

# The number of characters in the input language.
NUM_CHARS = 27
INPUT_LEN = 10

IN_CH_1 = NUM_CHARS
OUT_CH_1 = 400

IN_CH_2 = IN_CH_1 + OUT_CH_1
OUT_CH_2 = 400

IN_CH_3 = IN_CH_2 + OUT_CH_2
OUT_CH_3 = 400

IN_CH_4 = IN_CH_3 + OUT_CH_3
OUT_CH_4 = 400

IN_CH_5 = IN_CH_4 + OUT_CH_4
OUT_CH_5 = 400

# Use a fully-connected layer for the output, with no pooling.
IN_CH_OUT = INPUT_LEN * (IN_CH_5 + OUT_CH_5)
OUT_CH_OUT = NUM_CHARS


# A convolutional model for predicting the next letter.
class ConvLetterModel(nn.Module):
    def __init__(self):
        super(ConvLetterModel, self).__init__()

        self.conv_1 = nn.Conv1d(
            in_channels=IN_CH_1, out_channels=OUT_CH_1, kernel_size=5, padding=2
        )
        self.bn_1 = nn.BatchNorm1d(num_features=OUT_CH_1, momentum=0.01)
        self.act_1 = nn.ReLU()

        self.conv_2 = nn.Conv1d(
            in_channels=IN_CH_2, out_channels=OUT_CH_2, kernel_size=5, padding=2
        )
        self.bn_2 = nn.BatchNorm1d(num_features=OUT_CH_2, momentum=0.01)
        self.act_2 = nn.ReLU()

        self.conv_3 = nn.Conv1d(
            in_channels=IN_CH_3, out_channels=OUT_CH_3, kernel_size=5, padding=2
        )
        self.bn_3 = nn.BatchNorm1d(num_features=OUT_CH_3, momentum=0.00002)
        self.act_3 = nn.ReLU()

        self.conv_4 = nn.Conv1d(
            in_channels=IN_CH_4, out_channels=OUT_CH_4, kernel_size=5, padding=2
        )
        self.bn_4 = nn.BatchNorm1d(num_features=OUT_CH_4, momentum=0.00002)
        self.act_4 = nn.ReLU()

        self.conv_5 = nn.Conv1d(
            in_channels=IN_CH_5, out_channels=OUT_CH_5, kernel_size=5, padding=2
        )
        self.bn_5 = nn.BatchNorm1d(num_features=OUT_CH_5, momentum=0.00002)
        self.act_5 = nn.ReLU()

        self.out = nn.Linear(in_features=IN_CH_OUT, out_features=OUT_CH_OUT)

    # Initialize the model.
    def init_model(self):
        with torch.no_grad():
            init.orthogonal_(self.conv_1.weight)
            init.orthogonal_(self.conv_2.weight)
            init.orthogonal_(self.conv_3.weight)
            init.orthogonal_(self.conv_4.weight)
            init.orthogonal_(self.conv_5.weight)
            init.orthogonal_(self.out.weight)

    # Perform inference on x.
    def forward(self, x):
        # Do the convolution layers.
        c1 = self.conv_1(x)
        c1 = self.bn_1(c1)
        c1 = self.act_1(c1)
        x = torch.concat([x, c1], dim=1)

        c2 = self.conv_2(x)
        c2 = self.bn_2(c2)
        c2 = self.act_2(c2)
        x = torch.concat([x, c2], dim=1)

        c3 = self.conv_3(x)
        c3 = self.bn_3(c3)
        c3 = self.act_3(c3)
        x = torch.concat([x, c3], dim=1)

        c4 = self.conv_4(x)
        c4 = self.bn_4(c4)
        c4 = self.act_4(c4)
        x = torch.concat([x, c4], dim=1)

        c5 = self.conv_5(x)
        c5 = self.bn_5(c5)
        c5 = self.act_5(c5)
        x = torch.concat([x, c5], dim=1)

        # Flatten the input for the final layer.
        N = x.shape[0]
        x = x.reshape(N, -1)

        out = self.out(x)
        return out
