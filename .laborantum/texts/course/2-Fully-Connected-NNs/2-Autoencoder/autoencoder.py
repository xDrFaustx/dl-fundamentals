import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        ## YOUR CODE HERE

    def __call__(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE
        res = res.reshape(input_shape)
        return res


class Sampler(torch.nn.Module):
    def __init__(self, channels):
        ...
        ## YOUR CODE HERE


    def __call__(self, signal):
        res = signal
        mu = signal
        sigma = signal

        ## YOUR CODE HERE
        return res, mu, sigma


class VAE(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        ## YOUR CODE HERE

    def __call__(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE
        res = res.reshape(input_shape)
        return res, mu, sigma
