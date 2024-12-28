# AutoGrad

AutoGrad is a Python-based computational framework for building and training neural networks from scratch, inspired by libraries like PyTorch and TensorFlow. It leverages automatic differentiation to compute gradients efficiently, enabling users to train machine learning models using backpropagation. This project serves as both an educational tool and a lightweight framework for understanding the inner workings of neural networks and gradient-based optimization.

---

## Key Features
- **Automatic Differentiation**: AutoGrad tracks operations on scalar values in a computational graph, automatically computing gradients via backpropagation.
- **Custom Neural Network Components**:
  - **Neuron**: Represents a single neuron with weighted inputs, a bias, and a non-linear activation function (`tanh`).
  - **Layer**: A collection of neurons that processes input vectors and outputs activations.
  - **MLP (Multi-Layer Perceptron)**: A stack of layers that forms a complete feedforward neural network.
- **Lightweight and Modular**: Designed to be simple, modular, and easy to extend for educational purposes.

---

## How It Works

### Core Engine: `Value` Class
The `Value` class in `engine.py` represents a single scalar value in the computational graph. It supports:
- Arithmetic operations (+, -, *, /, **) with gradient tracking.
- Activation functions like `tanh` and `exp`.
- Backward propagation for gradient computation using the chain rule.

Example:
```python
from engine import Value

x = Value(2.0)
y = Value(3.0)
z = x * y + x**2
z.backward()

print(f"z: {z.data}, dz/dx: {x.grad}, dz/dy: {y.grad}")
```

### Neural Network Components: `nn.py`
The `nn.py` module provides building blocks for constructing neural networks:
1. **Neuron**: A single computational unit with weights, bias, and a `tanh` activation.
2. **Layer**: A collection of neurons.
3. **MLP**: A stack of layers forming a complete neural network.

Example:
```python
from nn import MLP

# Create an MLP with 3 inputs, 2 hidden layers, and 1 output
model = MLP(3, [4, 4, 1])

# Input data
x = [Value(1.0), Value(2.0), Value(3.0)]

# Forward pass
output = model(x)
print("Output:", output)
```

---

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/autograd.git
cd autograd
```

Install dependencies (if any).

---

## Usage

### Training a Neural Network
Define a model, compute the loss, and perform gradient descent:
```python
from nn import MLP
from engine import Value

# Create an MLP with 2 inputs and 1 output
n = MLP(3, [4,4,1])

# input values
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
    ]
ys = [1.0, -1.0, -1.0, 1.0] # desired values  

# Training loop
for k in range(100):
    # forward pass
    ypreds = [n(x) for x in xs]
    loss = sum((ygt - yout)**2 for (ygt, yout) in zip(ys, ypreds))
    
    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    for p in n.parameters():
        p.data += -0.05*p.grad
    
    print(f"Step: {k} | MSE Loss: {loss.data}")
```

---

## Why AutoGrad?
AutoGrad is perfect for:
- **Learning by Doing**: Understand the internals of backpropagation and neural network training.
- **Custom Projects**: Experiment with custom neural network architectures.
- **Research Prototyping**: Quickly test ideas without the overhead of large frameworks.

---

## Contributions
Contributions are welcome! Feel free to fork the repository and submit a pull request with your improvements or new features.

---

## License
This project is open-source and available under the MIT License.

---
