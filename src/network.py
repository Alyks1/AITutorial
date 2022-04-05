import numpy as np
import mnist_loader
import random

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight,a)+bias)
        return a
    
    def StochasticGradientDecent(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
        # training_data is list of tuples '(training_input,expected_output)'
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [ training_data[i:i+mini_batch_size] for i in range(0, n, mini_batch_size)]
            # learn
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            # test
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {epoch} complete")

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for actual, expected in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(actual,expected)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
    
    def backprop(self, actual, expected):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feed Forward
        activation = actual
        activations = [actual]
        zs = []

        # This is like feedforward fn but appends zs
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], expected) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(actual)), expect) for (actual,expect) in test_data]
        return sum(int(actual == expect) for (actual,expect) in test_results)
    
    def cost_derivative(self, output_activations, expected):
        return pow(output_activations - expected, 2)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784,16,16,10])
    net.StochasticGradientDecent(training_data, 30 , 10, 3.1415, test_data=test_data)