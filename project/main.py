import numpy as np
import pickle
from project.network import Network


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

mlp = Network(2, 4, True)
# train network
mlp.train(items_data, targets_data, 130, 0.3)

for j, inputs in enumerate(items_test):
    target = targets_test[j]
    mlp.forward(inputs)
    print(f"{mlp.layer_list[-1].output} wynik sieci\n {target} wynik wzorcowy")


def choose_menu():
    while True:
        function_type = input("""\nWybierz działanie z listy:
[1] - wyjdź z programu
[2] - stwórz sieć neuronową
[3] - zapisz sieć neuronową do pliku
[4] - wczytaj sieć neuronową z pliku
: """)
        if function_type not in ["1", "2", "3", "4"]:
            print("Wybierz opcję 1, 2, 3 lub 4")
        else:
            break
    return int(function_type)


run = True
mlp = None
while run:
    output_menu = choose_menu()
    if output_menu == 1:
        run = False
    elif output_menu == 2:
        nr_of_inputs = int(input("podaj ile będzie wejść do sieci: "))
        number = int(input("napisz 1 jeśli chcesz brać pod uwagę bias, 0 jeśli nie: "))
        is_bias_on = False
        if number == 1:
            is_bias_on = True
        nr_of_hidden_layers = int(input("podaj ilość warstw ukrytych: "))
        mlp = Network(nr_of_hidden_layers, nr_of_inputs, is_bias_on)
        print(mlp.weights)
        print(mlp.biases)
    elif output_menu == 3:
        filename = input("podaj nazwe pliku,do którego chcesz zapisać sieć, bez rozszerenia: ") + ".pkl"
        with open(filename, 'wb') as file:
            pickle.dump(mlp, file)
    elif output_menu == 4:
        filename = input("podaj nazwe pliku,z którego chcesz wczytać sieć, bez rozszerenia: ") + ".pkl"
        with open(filename, 'rb') as file:
            mlp = pickle.load(file)



# TODO uwzglednic biasy, zrobic cały mechanizm tegoo czy w ogole je uwzglednic czy nie
# TODO wszelkie zapisywanie stanu do sieci itd
# TODO zmienic inicjalizacje tych weights, derivatives itd
# TODO zmienic trening
# TODO ogarnac lepiej backpropagacje na odpowiedz
