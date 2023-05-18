# trzyma neurony, zajmuje się inicjalizacją neuronów
# obliczanie wyjścia danej warstwy
# propgacja błędu wstecz podczas nauki
import numpy as np
import matplotlib.pyplot as plt
from project.layer import Layer
from project.neuron import Neuron


# warstwa wyzsza musi miec tyle inputow co poprzednia outputow
# w warstwie MLP bedzie trzeba to dobrze przekazywac do kolejnych warstw
# warstwa ostatnia bedzie miala inna funkcje aktywacji chyba i ogolnie troche sie bedzie roznic
# wiec moze najlepiej zrobic jakas klase abstrakcyjna i z niej dziedziczyc
# nie mam batchingu danych narazie idk czy bedzie potrzebne
class HiddenLayer(Layer):
    def __init__(self, input_num: int, neuron_num: int):
        super().__init__()
        self.neuron_num = neuron_num
        self.input_num = input_num
        self.neurons = np.array([Neuron(input_num) for _ in range(neuron_num)])  # na razie tworze po prostu numpy array

    def forward(self, inputs):
        self.output = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.output


def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


# data = [1,2, 3, 4 , 5]
# layer = HiddenLayer(5, 4)
# print(layer.forward(data))


# X, y = spiral_data(100, 3)
#
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
# plt.show()
X = [1, 2, 3, 4]
layer = HiddenLayer(4, 3)
print(layer.forward(X))
