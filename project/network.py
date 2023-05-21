import numpy as np
from project.functions import sigmoid_derivative, mean_squared_error
from project.layer import Layer


class Network:
    def __init__(self, layer_num, input_num):
        self.layer_num = layer_num
        self.input_num = input_num
        self.layer_list = []
        inputs_layer = input_num
        for i in range(layer_num):
            neuron_num = int(input(f"Enter number of neurons in {i + 1} hidden layer: "))
            self.layer_list.append(Layer(inputs_layer, neuron_num))
            inputs_layer = neuron_num
        self.layer_list.append(Layer(inputs_layer, int(input("Enter number of neurons in output layer: "))))

        derivatives = [np.zeros((input_num, self.layer_list[0].neuron_num))]
        derivatives.extend([np.zeros((layer.neuron_num, self.layer_list[i + 1].neuron_num)) for i, layer in
                            enumerate(self.layer_list[:-1])])
        self.derivatives = derivatives

        activations = [np.zeros(input_num)]
        activations.extend([np.zeros(layer.neuron_num) for layer in self.layer_list])
        self.activations = activations

        weights = [np.zeros((input_num, self.layer_list[0].neuron_num))]
        weights.extend([np.zeros((layer.neuron_num, self.layer_list[i + 1].neuron_num)) for i, layer in
                        enumerate(self.layer_list[:-1])])
        self.weights = weights

    def forward(self, input_vector):
        self.activations[0] = np.array(input_vector)
        next_activation = self.activations[0]
        for x in range(len(self.layer_list)):
            next_activation = self.layer_list[x].forward(next_activation)
            self.activations[x + 1] = next_activation
            self.weights[x] = self.layer_list[x].get_weights()

    def back_propagation(self, error):
        for x in reversed(range(len(self.derivatives))):
            activation = self.activations[x + 1]
            delta = error * sigmoid_derivative(activation)
            delta_resh = delta.reshape(delta.shape[0], -1).T
            curr_act = self.activations[x]
            curr_act_resh = curr_act.reshape(curr_act.shape[0], -1)
            self.derivatives[x] = np.dot(curr_act_resh, delta_resh)

            error = np.dot(delta, self.weights[x])

    def update_weights(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights -= derivatives.T * learning_rate

    def update_neurons_weights(self):
        for x in range(len(self.layer_list)):
            self.layer_list[x].update_weights(self.weights[x])

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_errors = 0
            for j, input in enumerate(inputs):
                target = targets[j]

                self.forward(input)

                error = self.activations[-1] - target  # to bede outputy sieci

                self.back_propagation(error)
                self.update_weights(learning_rate)
                self.update_neurons_weights()

                sum_errors += mean_squared_error(target, self.activations[-1])

            print(f"Error: {sum_errors / len(inputs)} at epoch {i + 1}")


if __name__ == "__main__":
    data = []
    with open('Data/iris.csv', 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                values = line.split(',')
                features = [float(value) for value in values[:4]]
                if values[4] == 'Iris-setosa':
                    target = [1, 0, 0]
                if values[4] == 'Iris-versicolor':
                    target = [0, 1, 0]
                if values[4] == 'Iris-virginica':
                    target = [0, 0, 1]
                data.append(features + target)

    data = np.array(data)

    training_data = data[:130]
    test_data = data[130:]

    items_data = training_data[:, :4]
    targets_data = training_data[:, 4:7]

    items_test = test_data[:, :4]
    targets_test = test_data[:, 4:7]

    # create a Multilayer Perceptron with one hidden layer
    mlp = Network(2, 4)

    # train network
    mlp.train(items_data, targets_data, 130, 0.3)

    for j, inputs in enumerate(items_test):
        target = targets_test[j]
        mlp.forward(inputs)
        print(f"{mlp.layer_list[-1].output} wynik sieci\n {target} wynik wzorcowy")

# TODO uwzglednic biasy, zrobic cały mechanizm tegoo czy w ogole je uwzglednic czy nie
# TODO wszelkie zapisywanie stanu do sieci itd
# TODO zmienic inicjalizacje tych weights, derivatives itd
# TODO zmienic trening
# TODO ogarnac lepiej backpropagacje na odpowiedz
