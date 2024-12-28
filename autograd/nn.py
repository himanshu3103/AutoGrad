import random
from engine import Value

class Module:
    """
    Base class for all neural network components. Provides utility functions for managing gradients
    and parameters. Can be inherited by other classes to define specific modules (e.g., Neuron, Layer).
    """
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []
    

class Neuron(Module):
    """
    Represents a single neuron in a neural network. Each neuron has a set of weights and a bias.
    The neuron computes a weighted sum of its inputs followed by a non-linear activation function (tanh).
    """
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        # print(f"weights: {self.w}")
        self.b = Value(random.uniform(-1,1)) 
        # print(f"bias: {self.b}")

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # second parameter is start
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"Neuron ({len(self.w)})"
    
class Layer(Module):
    """
    Represents a layer of neurons in a neural network. Each layer contains multiple neurons
    and processes a set of inputs to produce a set of outputs.
    """
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        # print(f"neurons in the layer: {self.neurons}")
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):    
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):
    """
    Represents a multi-layer perceptron (MLP), which is a sequence of layers of neurons.
    Processes inputs through all layers sequentially.
    """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        # print(f"layers of neurons: {self.layers}")

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"