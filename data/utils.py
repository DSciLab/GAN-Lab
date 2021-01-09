from torch import nn


class LinearNormalize(nn.Module):
    def forward(self, x):
        return (x - x.min()) / x.max()
