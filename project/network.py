from project.hidden_layer import HiddenLayer
from project.output_layer import OutputLayer


class Network:
    def __init__(self, layer_num, input_num):
        self.layer_num = layer_num
        self.input_num = input_num
        self.layer_list = []
        inputs = input_num
        for i in range(layer_num):
            neuron_num = int(input(f"Enter number of neurons in {i + 1} hidden layer: "))
            self.layer_list.append(HiddenLayer(inputs, neuron_num))
            inputs = neuron_num
        self.layer_list.append(OutputLayer(inputs, int(input("Enter number of neurons in output layer: "))))



network = Network(2, 3)
data = [1,2, 3]
forward1 = network.layer_list[0].forward(data)
forward2 = network.layer_list[1].forward(forward1)
output = network.layer_list[2].forward(forward2)
print(output)



