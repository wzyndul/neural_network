from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.neuron_num = None
        self.input_num = None
        self.neurons = None
        self.output = None

    @abstractmethod
    def forward(self, inputs):  # idk czy potrzebuje forwarda, ja kto tylko w hidden bedzie
        pass