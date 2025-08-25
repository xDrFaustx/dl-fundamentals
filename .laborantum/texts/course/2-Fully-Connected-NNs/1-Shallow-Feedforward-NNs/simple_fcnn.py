import torch

class SimpleFCNN(torch.nn.Module):
    def __init__(
            self, 
            channels=[],
            n_classes=10,
            activation=torch.nn.ReLU):
        ...
        ## YOUR CODE HERE

    def __call__(self, signal):
        res = signal.reshape([signal.shape[0], -1])
        ## YOUR CODE HERE
        return res