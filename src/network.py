import numpy as np
import os

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.rand(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight,a)+bias)
        return a
    
    def StochasticGradientDecent(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
        # training_data is list of tuples '(training_input,expected_output)'
        n = len(training_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[i:i+mini_batch_size]
                for i in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {epoch} complete")

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backdrop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def backdrop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activation[-1],y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-1]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-1+1].transpose(), delta) * sp
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-1-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))