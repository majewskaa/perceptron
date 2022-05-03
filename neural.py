#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:51:50 2021

@author: Rafał Biedrzycki
Kodu tego mogą używać moi studenci na ćwiczeniach z przedmiotu Wstęp do Sztucznej Inteligencji.
Kod ten powstał aby przyspieszyć i ułatwić pracę studentów, aby mogli skupić się na algorytmach sztucznej inteligencji.
Kod nie jest wzorem dobrej jakości programowania w Pythonie, nie jest również wzorem programowania obiektowego, może zawierać błędy.

Nie ma obowiązku używania tego kodu.
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


p = [2,9]
np.random.seed(1)

def f(x):
    return np.sin(x * np.sqrt(p[0] + 1)) + np.cos(x * np.sqrt(p[1] + 1))

# f logistyczna jako przykĹad sigmoidalej
def sigmoid(x):
    return 1/(1+np.exp(-x))

#pochodna fun. 'sigmoid'
def d_sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s * (1-s)

#f. straty
def nloss(y_out, y):
    return (y_out - y) ** 2

#pochodna f. straty
def d_nloss(y_out, y):
    return 2*( y_out - y )

class DlNet:
    def __init__(self, num_of_hidden_neurons=9, learning_rate=0.1):
        self.y_out = 0
        self.hidden_layer_size = num_of_hidden_neurons
        self.init_weights_and_biases()
        self.learning_rate = float(learning_rate)

        self.hidden_layer_inputs = None
        self.hidden_layer_activations = None

    def init_weights_and_biases(self):
        self.weights = [
            np.random.uniform(-1.0, 1.0, size=(self.hidden_layer_size, 1)),
            np.random.uniform(-1.0, 1.0, size=(1, self.hidden_layer_size))]
        self.biases = [np.zeros((self.hidden_layer_size, 1)), np.zeros((1, 1))]

    def forward(self, x):
        self.hidden_layer_inputs = np.dot(self.weights[0], x) + self.biases[0]
        self.hidden_layer_activations = sigmoid(self.hidden_layer_inputs)
        self.y_out = (np.dot(self.weights[-1], self.hidden_layer_activations) + self.biases[-1]).item()

    def predict(self, x):
        self.forward(x)
        return self.y_out

    def train_SGD(self, x_set, y_set, epochs, mini_batch_size):
        training_points = [(x, y) for x, y in zip(x_set, y_set)]
        for _ in range(epochs):
            np.random.shuffle(training_points)
            batches = np.split(np.array(training_points), len(training_points) / mini_batch_size)
            for batch in batches:
                self.update(batch)

    def update(self, batch):
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]

        for x, y in batch:
            d_nabla_b, d_nabla_w = self.backward(x, y)

            # calculate sum of gradients from all the batches
            nabla_w[-1] += d_nabla_w[-1]
            nabla_w[0] += d_nabla_w[0]
            nabla_b[-1] += d_nabla_b[-1]
            nabla_b[0] += d_nabla_b[0]

        # in this section we divide the calculated sums by the number of samples
        # in the batch argument getting the gradient which fits 'all' the training
        # points istead of just one
        self.weights[-1] -= self.learning_rate * (nabla_w[-1] / len(batch))
        self.weights[0] -= self.learning_rate * (nabla_w[0] / len(batch))
        self.biases[-1] -= self.learning_rate * (nabla_b[-1] / len(batch))
        self.biases[0] -= self.learning_rate * (nabla_b[0] / len(batch))

    def backward(self, x, y):
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]

        self.forward(x)

        # from cost function to output layer
        delta = d_nloss(self.y_out, y)  # activation function at last layer is just f(x) = x so its derivative = 1
        nabla_b[-1] = np.array(delta).reshape((1, 1))
        nabla_w[-1] = self.hidden_layer_activations.transpose() * delta

        # from output layer to the hidden layer
        delta = self.weights[-1].transpose() * d_sigmoid(self.hidden_layer_inputs) * delta
        nabla_b[0] = delta
        nabla_w[0] = delta * x

        return (nabla_b, nabla_w)



def train_gif(self, x_set, y_set, iters=10, num_of_batches=1):
        training_data = np.c_[x_set, y_set]
        filenames_gif = []
        for i in range(iters):
            np.random.shuffle(training_data)
            batches = np.split(training_data, num_of_batches)

            for batch in batches:
                self.update(batch)

            yh = [self.predict(xn) for xn in x_set]

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            plt.plot(x_set, y_set, 'r')
            plt.plot(x_set, yh, 'b')
            f_name = f'{i}.png'
            plt.savefig(f_name)
            plt.close()
            filenames_gif.append(f_name)

        with imageio.get_writer('mygif.gif', mode='I') as writer:
            for filename in filenames_gif:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(filenames_gif):
            os.remove(filename)