import numpy as np
from project.layer import Layer
from project.neuron import Neuron


# Te info to jakieś moje notatki do implementacji nie czytaj tego - nic przydatnego
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
        self.neurons = np.array([Neuron(input_num) for _ in range(neuron_num)])
        # tworze numpy array w którym znajdują sie kolejne to neurony

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.output
        # dla każdego neuronu w mojej tablicy wywołuje funkcjie forward
