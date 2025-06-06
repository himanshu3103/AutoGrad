import math

class Value:
    
    """
    The Value class represents a scalar value in a computational graph for automatic differentiation.
    It supports operations like addition, multiplication, exponentiation, and more, while tracking
    gradients for backpropagation. Each Value object stores its value, gradient, and references to
    its parent nodes in the graph. The `backward` method computes gradients using the chain rule.
    """

    def __init__(self, data, _children = (), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __add__(self, other):
        other = other if isinstance(other,Value) else Value(other) # this functionality allows us to directly perform operation with non Value obejcts by explicitly first converting it to Value object
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, {self, other}, '*') 
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, other): # self - other
        return self + (-other)

    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f'**{other}')
        def _backward():
            self.grad += (other) * (self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        vis = set()
        def build_topo(v):
            if v not in vis:
                vis.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
            return
        self.grad = 1.0
        build_topo(self)
        for node in reversed(topo):
            node._backward()
    
    def __repr__(self):
        return f"Value(data={self.data})"

    def __rmul__(self, other): # other * self
        return self * other
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __rtruediv__(self, other): # other / self
        return other * (self ** -1)
    def __neg__(self): # -self
        return self * -1
    
