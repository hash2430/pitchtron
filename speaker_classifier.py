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

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        return -1 * grad_output * x[0]

def gradient_reversal_layer(x):
    grl = GradientReversalLayer.apply
    return grl(x)