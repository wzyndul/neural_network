# trzyma neurony, zajmuje się inicjalizacją neuronów
# obliczanie wyjścia danej warstwy
# propgacja błędu wstecz podczas nauki
import numpy as np

from project.neuron import Neuron


class Layer:
    def __init__(self, neuron_num: int):
        self.neuron_num = neuron_num
        self.neurons = np.array([Neuron() for _ in range(neuron_num)])  # na razie tworze po prostu numpy array

    def forward(self, inputs):
        outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])  # robi lise i konwertuje na np.array
        return outputs
