import numpy as np
from project.activation_functions import sigmoid_derivative, mean_squared_error
from project.layer import Layer


class Network:
    def __init__(self, layer_num, input_num):
        self.layer_num = layer_num
        self.input_num = input_num
        self.layer_list = []
        inputs = input_num
        for i in range(layer_num):
            neuron_num = int(input(f"Enter number of neurons in {i + 1} hidden layer: "))
            self.layer_list.append(Layer(inputs, neuron_num))
            inputs = neuron_num
        self.layer_list.append(Layer(inputs, int(input("Enter number of neurons in output layer: "))))


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

        # return error

    def update_weights(self, learning_rate):
        """Learns by descending the gradient
        Args:
            learning_rate (float): How fast to learn.
        """
        # update the weights and biases by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights -= derivatives.T * learning_rate

    def update_neurons_weights(self):
        for x in range(len(self.layer_list)):
            self.layer_list[x].update_weights(self.weights[x])


    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model by running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0
            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                self.forward(input)

                error = self.activations[-1] - target # to bede outputy sieci

                self.back_propagation(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights and biases)
                self.update_weights(learning_rate)
                self.update_neurons_weights()

                # keep track of the MSE for reporting later
                sum_errors += mean_squared_error(target, self.activations[-1])

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(inputs), i + 1))

        print("Training complete!")
        print("=====")



#

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


    items = training_data[:, :4]
    targets = training_data[:, 4:7]

    items_test = test_data[:, :4]
    targets_test = test_data[:, 4:7]

    # create a Multilayer Perceptron with one hidden layer
    mlp = Network(2, 4)

    # train network
    mlp.train(items, targets, 130, 0.3)



    for j, input in enumerate(items_test):
        target = targets_test[j]
        mlp.forward(input)
        print(f"{mlp.layer_list[-1].output} wynik sieci\n {target} wynik wzorcowy")



#TODO uwzglednic biasy, zrobic cały mechanizm tegoo czy w ogole je uwzglednic czy nie
#TODO wszelkie zapisywanie stanu do sieci itd
#TODO zmienic inicjalizacje tych weights, derivatives itd
#TODO zmienic trening
#TODO ogarnac lepiej backpropagacje na odpowiedz