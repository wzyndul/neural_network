from abc import ABC, abstractmethod

# abstrakcyjna klasa z której dziedziczy hidden layer i output layer
class Layer(ABC):
    def __init__(self):
        self.neuron_num = None
        self.input_num = None
        self.neurons = None
        self.output = None

    @abstractmethod
    def forward(self, inputs):
        pass
