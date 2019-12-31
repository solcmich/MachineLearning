import random

import numpy as np

DEBUG = True


class NeuralNetwork:

    def __init__(self, data, layers, epochs, batch_size, learning_rate):
        self.input_layer = layers[-1]
        self.output_layer = layers[0]
        self.layers = layers
        self.biases = np.array([np.random.randn(y, 1) for y in layers[1:]])
        self.weights = np.array([np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])])
        self.train_data, self.evaluation_data, self.test_data = self.split_data(data, [6, 2, 2])
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def split_data(self, data, ratios):
        data_len = len(data)
        train_data = data[0:220]
        evaluation_data = data[220: 280]
        test_data = data[280:]
        return train_data, evaluation_data, test_data

    def set_params_random(self, epochs_range, batch_size_range, layers_range, neurons_in_layer_range,  learning_rate):
        self.epochs = random.randrange(epochs_range[0], epochs_range[1])

        self.batch_size = random.randrange(batch_size_range[0], batch_size_range[1])
        #layers
        hidden_layers_num = random.randrange(layers_range[0], layers_range[1])
        self.layers = [self.output_layer]
        for i in range(hidden_layers_num):
            self.layers.append(random.randrange(neurons_in_layer_range[0], neurons_in_layer_range[1]))
        self.layers.append(self.input_layer)

        self.biases = np.array([np.random.randn(y, 1) for y in self.layers[1:]])
        self.weights = np.array([np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])])

        f_range = np.arange(learning_rate[0], learning_rate[1], learning_rate[2])
        index = random.randrange(0, len(f_range) - 1)
        self.learning_rate = f_range[index]

        return self.epochs, self.batch_size, self.layers, self.learning_rate

    def set_params(self, epochs, batch_size, layers, learning_rate):
        self.epochs = epochs

        self.batch_size = batch_size

        self.layers = layers

        self.biases = np.array([np.random.randn(y, 1) / 10 for y in self.layers[1:]])
        self.weights = np.array([np.random.randn(y, x) / 10 for x, y in zip(self.layers[:-1], self.layers[1:])])

        self.learning_rate = learning_rate

    def train(self, guesses, epochs_range, batch_size_range, layers_range, neurons_in_layer_range,
              learning_rate_range):
        self.stochastic_gradient_descent(self.train_data)
        accuracy = round(self.accuracy(self.evaluation_data), 3)
        dct = {accuracy: (self.epochs, self.batch_size, self.layers, self.learning_rate)}
        for i in range(guesses):
            if DEBUG:
                print('******************************************')
                val = dct[accuracy]
                print(f'Tuning params after {i + 1} is {round (accuracy * 100, 2)} % accurate'
                    f' with params: epochs {val[0]} batch: {val[1]}, layers: '
                    f'{val[2]}, learning_rate: {val[3]}')
                print('******************************************')

            self.set_params_random(epochs_range, batch_size_range, layers_range, neurons_in_layer_range,
                                   learning_rate_range)
            self.stochastic_gradient_descent(self.train_data)
            accuracy = self.accuracy(self.evaluation_data)
            dct[accuracy] = (self.epochs, self.batch_size, self.layers, self.learning_rate)
        keys = dct.keys()
        max_acc = max(keys)
        val = dct[max_acc]

        self.set_params(val[0], val[1], val[2], val[3])
        self.stochastic_gradient_descent(self.train_data)

        print(f'Max accuracy is: {max_acc} for vals: epochs {val[0]} batch: {val[1]}, layers: '
              f'{val[2]}, learning_rate: {val[3]}')

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sgm(w@a + b)
        return a

    def stochastic_gradient_descent(self, data):
        n = len(data)
        if DEBUG:
            print('***********************************************************')
        for j in range(self.epochs):
            if DEBUG:
                acc = self.accuracy(self.train_data)
                print(f'Accuracy after {j} epochs: { round (acc, 2) * 100 } %')
            batches = [data[k:k + self.batch_size] for k in range(0, n, self.batch_size)]
            for b in batches:
                self.update(b)
            self.learning_rate = self.learning_rate * 1/(1+0.00001 * j)
        if DEBUG:
            print('***********************************************************')

    def update(self, batch):
        # update weights and biases by applying SDG. Nabla stands for "reverse" triangle
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        l = ([row[:-1] for row in batch])
        r = ([row[-1] for row in batch])

        # go through all data in mini_batch
        for x, y in zip(l, r):
            d_nabla_b, d_nabla_w = self.back_propagation(x, y)
            # update by computed gradient vector
            nabla_b = [nb + nbb for nb, nbb in zip(nabla_b, d_nabla_b)]
            nabla_w = [nw + nbw for nw, nbw in zip(nabla_w, d_nabla_w)]
        # apply final step in gradient descent
        b_len = len(batch)
        self.biases = [b - (self.learning_rate / b_len) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (self.learning_rate / b_len) * nw for w, nw in zip(self.weights, nabla_w)]

    def back_propagation(self, x, y):
        # init
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # input layer activation
        activation = np.array(x).reshape(len(x), 1)
        # store all activations
        activations = [activation]
        # init array for z vectors -> neuron values on each layer
        z_vecs = []
        # simple feed forward
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_vecs.append(z)
            activation = self.sgm(z)
            activations.append(activation)
        # now the magic -> backward pass
        # compute first delta from last activation and last
        delta = self.cost_derivate(activations[-1], y) * self.sgm_derivate(z_vecs[-1])
        # for biases, last derivative is constant 1, so just
        nabla_b[-1] = delta
        # for weights, dot it with activations from last layer
        nabla_w[-1] = np.dot(delta, np.transpose(activations[-2]))

        # repeat the process
        for i in range(2, len(self.layers)):
            # obtain z vec from back to front
            z = activations[-i]
            sd = self.sgm_derivate(z)
            # same process as above
            delta = np.dot(np.transpose(self.weights[-i + 1]), delta) * sd
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, np.transpose(activations[-i - 1]))

        return nabla_b, nabla_w

    def cost_derivate(self, x, y):
        return (x - y)

    def sgm_derivate(self, x):
        return self.sgm(x) * (1 - self.sgm(x))

    @staticmethod
    def sgm(x):
        return 1.0 / (1.0 + np.exp(-x))

    def accuracy(self, data):
        l = ([row[:-1] for row in data])
        r = ([row[-1] for row in data])
        cnt = 0
        for i in range(len(l)):
            arr = np.array(l[i]).reshape(1, self.layers[0])
            arr = arr.reshape(self.layers[0], 1)
            res = self.feed_forward(arr)
            if abs(r[i] - res[0][0]) < .5:
                cnt = cnt + 1

        l_l = len(l)
        ret = float(cnt) / float(l_l)
        return ret
