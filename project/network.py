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

    def next_d(self, i, k):
        if i == len(self.layer_list) - 1:
            loss = self.layer_list[-1].loss(k)
            return loss
        else:
            result = 0
            for j in range(self.layer_list[i + 1].neuron_num):
                wagi = self.layer_list[i + 1].neurons[j].weights[j]
                sigmoida = sigmoid_derivative(self.layer_list[i + 1].neurons[j].weighted_sum)
                next = self.next_d(i + 1, k)
                # result += self.layer_list[i + 1].neurons[j].weighted_sum * sigmoid_derivative(
                #     self.layer_list[i + 1].neurons[j].weighted_sum) * self.next_d(i + 1, k)
                result += wagi * sigmoida * next
            return result

    def set_inputs(self, inputs):
        self.inputs = inputs

    def calculate_gradients(self):
        gradient_list = []
        for i in range(len(self.layer_list) - 1, -1, -1):  # im going from last index to first
            layer = self.layer_list[i]
            lower_layer = self.layer_list[i - 1]
            # warunek jakis bo jak dojde do 0 indexu to wtedy bede operowal inaczej troche
            if i == 0:
                for j in range(layer.neuron_num):
                    for k in range(len(self.inputs)):
                        pierwsza_czesc = self.inputs[k] * sigmoid_derivative(layer.neurons[j].weighted_sum)
                        druga_czesc = self.next_d(i, k)
                        # gradient_list.append(self.inputs[k] * sigmoid_derivative(layer.neurons[j].weighted_sum) * self.next_d(i, k))
                        gradient_list.append(pierwsza_czesc *  druga_czesc)

            else:
                for j in range(layer.neuron_num):
                    for k in range(lower_layer.neuron_num):
                        pierwsza_czesc = lower_layer.output[k] * sigmoid_derivative(layer.neurons[j].weighted_sum)
                        druga_czesc = self.next_d(i,k)
                        # gradient_list.append(lower_layer.output[k] * sigmoid_derivative(layer.neurons[j].weighted_sum) * self.next_d(i,k))
                        gradient_list.append(pierwsza_czesc * druga_czesc)
        return gradient_list


def podziel_na_3_tablice(new_weights):
    num_tables = len(new_weights) // 3  # czy to atomowe?????
    tables = []
    for i in range(num_tables):
        start_index = i * 3
        end_index = start_index + 3
        table = new_weights[start_index:end_index]
        tables.append(table)
    return tables



etykiety = ["random", "random2", "random3", "width", "name"]
iris_data = pd.read_csv('Data/iris.csv', sep=',', names=etykiety)

# Extract the 4th column ("width") and 5th column ("name") from iris_data
width_column = iris_data["width"].values
name_column = iris_data["name"].values

# Create a new column based on conditions
new_column = np.where(np.logical_or(name_column == "Iris-versicolor", name_column == "Iris-virginica"), 0, 1)

# Create the NumPy array with the desired columns
result_array = np.column_stack((width_column, new_column))

training_data = result_array[:100] # 100 pierwszych


test_data = result_array[:50] # 50 ostatnich

learning_rate = 0.6
momentum = 0.6
network = Network(1, 1)


# np.random.shuffle(result_array)

SGTALA = 200
for dupa in range(SGTALA):

    sredni_koszt = 0

    np.random.shuffle(training_data)
    for x in training_data:
        table1 = [x[0]]
        table2 = [x[1]]
    #     table1 = [0.2]
    #     table2 = [1]

        network.set_inputs(table1)
        for net in network.layer_list:
            net.set_true_values(table2)

        network.set_inputs(table1)
        forward1 = network.layer_list[0].forward(table1)
        output = network.layer_list[1].forward(forward1)

        gradient = network.calculate_gradients()


    # trzy_tablice = podziel_na_3_tablice(gradient)
    #print(gradient)



    # network.layer_list[2].update_weights(trzy_tablice[0], learning_rate)
    # network.layer_list[1].update_weights(trzy_tablice[1], learning_rate)
    # network.layer_list[0].update_weights(trzy_tablice[2], learning_rate)

        network.layer_list[1].update_weights(gradient[0], learning_rate, momentum)
        network.layer_list[0].update_weights(gradient[1], learning_rate, momentum)

        sredni_koszt += network.layer_list[1].mean_loss()
        # if dupa == SGTALA - 1:
        #     print(f"output {output} desired output {table2}")

    if dupa == SGTALA - 1 or dupa == 0:
        # sorted_array = result_array[result_array[:, 1].argsort()]
        # print(str(output) + " " + str(table2))
        print(f"sredni koszt {sredni_koszt / 150}")
        # print(f"output {output} desired output {table2}")


            # print(f"sredni koszt {sredni_koszt}")
            # print(f"output {output}")




#
# for x in test_data:
#     table1 = [x[0]]
#     table2 = [x[1]]
#
#     network.set_inputs(table1)
#     for net in network.layer_list:
#         net.set_true_values(table2)
#
#     network.set_inputs(table1)
#     forward1 = network.layer_list[0].forward(table1)
#     output = network.layer_list[1].forward(forward1)



