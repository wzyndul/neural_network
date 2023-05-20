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

    def update_weights(self, new_weights, learning_rate):
        num_tables = len(new_weights) // self.neuron_num # czy to atomowe?????
        tables = []

        for i in range(num_tables):
            start_index = i * self.neuron_num
            end_index = start_index + self.neuron_num
            table = new_weights[start_index:end_index]
            tables.append(table)

        for neuron, table in zip(self.neurons, tables):
            neuron.update_weight(learning_rate, table)





        pass

    @abstractmethod
    def forward(self, inputs):
        pass
