import numpy as np
import pandas as pd

from project.activation_functions import softmax_derivative, sigmoid_derivative
from project.hidden_layer import HiddenLayer
from project.output_layer import OutputLayer


class Network:
    def __init__(self, layer_num, input_num):
        self.layer_num = layer_num
        self.input_num = input_num
        self.layer_list = []
        self.inputs = []
        inputs = input_num
        for i in range(layer_num):
            neuron_num = int(input(f"Enter number of neurons in {i + 1} hidden layer: "))
            self.layer_list.append(HiddenLayer(inputs, neuron_num))
            inputs = neuron_num
        self.layer_list.append(OutputLayer(inputs, int(input("Enter number of neurons in output layer: "))))

        weights = []
        w = np.zeros((input_num, self.layer_list[0].neuron_num))
        weights.append(w)
        for i in range(len(self.layer_list) - 1):
            w = np.zeros((self.layer_list[i].neuron_num, self.layer_list[i + 1].neuron_num))
            weights.append(w)
        self.weights = weights

        derivatives = []
        d = np.zeros((input_num, self.layer_list[0].neuron_num))
        derivatives.append(d)
        for i in range(len(self.layer_list) - 1):
            d = np.zeros((self.layer_list[i].neuron_num, self.layer_list[i + 1].neuron_num))
            derivatives.append(d)
        self.derivatives = derivatives

        activations = []
        a = np.zeros(input_num)
        activations.append(a)
        for i in range(len(self.layer_list)):
            a = np.zeros(self.layer_list[i].neuron_num)
            activations.append(a)
        self.activations = activations

    def forward(self, inputs):
        self.activations[0] = np.array(inputs)
        next_activation = self.activations[0]
        for x in range(len(self.layer_list)):
            next_activation = self.layer_list[x].forward(next_activation)
            self.activations[x + 1] = next_activation
            # print(self.layer_list[x].get_weights())
            self.weights[x] = self.layer_list[x].get_weights()

    def back_propagation(self, error):
        for x in reversed(range(len(self.derivatives))):
            activation = self.activations[x + 1]
            delta = error * sigmoid_derivative(activation)
            delta_resh = delta.reshape(delta.shape[0], -1).T
            curr_act = self.activations[x]
            curr_act_resh = curr_act.reshape(curr_act.shape[0], -1)
            self.derivatives[x] = np.dot(curr_act_resh, delta_resh)

            error = np.dot(delta, self.weights[x].T)

        return error


    def gradient_descent(self, learning_rate):
        """Learns by descending the gradient
        Args:
            learning_rate (float): How fast to learn.
        """
        # update the weights and biases by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def update_weights(self):
        pass














    def set_inputs(self, inputs):
        self.inputs = inputs




etykiety = ["random", "random2", "random3", "width", "name"]
iris_data = pd.read_csv('Data/iris.csv', sep=',', names=etykiety)

# Extract the 4th column ("width") and 5th column ("name") from iris_data
width_column = iris_data["width"].values
name_column = iris_data["name"].values

# Create a new column based on conditions
new_column = np.where(np.logical_or(name_column == "Iris-versicolor", name_column == "Iris-virginica"), 0, 1)

# Create the NumPy array with the desired columns
result_array = np.column_stack((width_column, new_column))

training_data = result_array[:100]  # 100 pierwszych

test_data = result_array[:50]  # 50 ostatnich

learning_rate = 0.6
momentum = 0.6
network = Network(1, 3)
data = [0.2, 0.1, 0.3]

# network.set_inputs(data)
# forward1 = network.layer_list[0].forward(data)
# output = network.layer_list[1].forward(forward1)
# print(output)

network.forward(data)
print(network.weights)
network.back_propagation([0.34, 0.32, 0.23])
network.gradient_descent(0.6)
print("network:")
print(network.weights)

# np.random.shuffle(result_array)

# SGTALA = 2
# for dupa in range(SGTALA):
#
#     sredni_koszt = 0
#
#     np.random.shuffle(training_data)
#     for x in training_data:
#         table1 = [x[0]]
#         table2 = [x[1]]
#         #     table1 = [0.2]
#         #     table2 = [1]
#
#         network.set_inputs(table1)
#         for net in network.layer_list:
#             net.set_true_values(table2)
#
#         network.set_inputs(table1)
#         forward1 = network.layer_list[0].forward(table1)
#         output = network.layer_list[1].forward(forward1)
#
#         gradient = network.calculate_gradients()
#
#         # trzy_tablice = podziel_na_3_tablice(gradient)
#         # print(gradient)
#
#         # network.layer_list[2].update_weights(trzy_tablice[0], learning_rate)
#         # network.layer_list[1].update_weights(trzy_tablice[1], learning_rate)
#         # network.layer_list[0].update_weights(trzy_tablice[2], learning_rate)
#
#         network.layer_list[1].update_weights(gradient[0], learning_rate, momentum)
#         network.layer_list[0].update_weights(gradient[1], learning_rate, momentum)
#
#         sredni_koszt += network.layer_list[1].mean_loss()
#         # if dupa == SGTALA - 1:
#         #     print(f"output {output} desired output {table2}")
#
#     if dupa == SGTALA - 1 or dupa == 0:
#         pass
#         # sorted_array = result_array[result_array[:, 1].argsort()]
#         # print(str(output) + " " + str(table2))
#         # print(f"sredni koszt {sredni_koszt / 150}")
#         # print(f"output {output} desired output {table2}")
#
#         # print(f"sredni koszt {sredni_koszt}")
#         # print(f"output {output}")
