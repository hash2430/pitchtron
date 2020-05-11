import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, text_embedding_dim, hidden_dim, num_speakers):
        super(Classifier, self).__init__()
        self.hidden = nn.Linear(text_embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_speakers)

    def forward(self, x):
        x = self.hidden(x)
        x = nn.functional.relu(x)
        x = self.output(x)
        return x

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x):
        return x

    def backward(self, grad_output):
        return -1 * grad_output

def gradient_reversal_layer(x):
    grl = GradientReversalLayer()
    return grl(x)