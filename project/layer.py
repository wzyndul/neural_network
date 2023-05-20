from abc import ABC, abstractmethod


# abstrakcyjna klasa z której dziedziczy hidden layer i output layer
class Layer(ABC):
    def __init__(self):
        self.neuron_num = None
        self.input_num = None
        self.neurons = None
        self.output = None
        self.true_values = None
        self.inputs = None

    def set_true_values(self, values):
        self.true_values = values

    def weighted_sum(self):
        return [neuron.weighted_sum for neuron in self.neurons]

    @abstractmethod
    def forward(self, inputs):
        pass
