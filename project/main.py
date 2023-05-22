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

# to sa te dane pomieszane ze tak powiem
training_data = []
test_data = []

# Iterate over the rows in data
for i, row in enumerate(data):
    # Determine the destination based on the index
    if i % 20 < 10:  # First 10 rows go to training data
        training_data.append(row)
    else:  # Next 10 rows go to test data
        test_data.append(row)

# Convert the lists to NumPy arrays
training_data = np.array(training_data)
test_data = np.array(test_data)

# len training data = 80
np.savetxt('data.txt', training_data)
# training_data = data[:130]
# test_data = data[130:]
#
# mlp = Network(2, 4, True)
# mlp.train(training_data, 300, 0.3)

# for sample in test_data:
#     target = sample[4:]
#     mlp.forward(sample[:4])
#     print(f"{mlp.layer_list[-1].output} wynik sieci\n {target} wynik wzorcowy")


def choose_menu():
    while True:
        function_type = input("""\nWybierz działanie z listy:
[1] - wyjdź z programu
[2] - stwórz sieć neuronową
[3] - zapisz sieć neuronową do pliku
[4] - wczytaj sieć neuronową z pliku
[5] - trenuj sieć 
[6] - testuj sieć
: """)
        if function_type not in ["1", "2", "3", "4", "5", "6"]:
            print("Wybierz opcję 1, 2, 3, 4, 5 lub 6.")
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
        is_bias_on = False
        if int(input("napisz 1 jeśli chcesz brać pod uwagę bias, 0 jeśli nie: ")) == 1:
            is_bias_on = True
        nr_of_hidden_layers = int(input("podaj ilość warstw ukrytych: "))
        mlp = Network(nr_of_hidden_layers, nr_of_inputs, is_bias_on)
    elif output_menu == 3:
        filename = input("podaj nazwe pliku,do którego chcesz zapisać sieć, bez rozszerenia: ") + ".pkl"
        with open(filename, 'wb') as file:
            pickle.dump(mlp, file)
    elif output_menu == 4:
        filename = input("podaj nazwe pliku,z którego chcesz wczytać sieć, bez rozszerenia: ") + ".pkl"
        with open(filename, 'rb') as file:
            mlp = pickle.load(file)
    elif output_menu == 5:
        which_data_set = int(input("napisz 1 - zbiór irysów, 2 - autoasocjacja, 3 - własny zbiór: "))
        data_for_training = None
        if which_data_set == 1:
            data_for_training = training_data  # trzeba odpowiednio wczytac
        elif which_data_set == 2:
            pass  # TODO dane z drugiego zadania
        else:
            data_for_training = np.loadtxt(input("podaj nazwę pliku z rozszerzeniem txt: "))
        nr_of_epochs = int(input("podaj liczbę epok: "))
        learning_rate = float(input("podaj prędkość nauki: "))
        error_level = float(input("podaj poziom błedu jaki chcesz osiągnąć: "))
        jump = int(input("podaj co ile epok mają być rejestrowane statystyki: "))
        shuffle = False
        if int(input("napisz 1 jeśli chcesz mieszać kolejności prezentacji wzorców: ")) == 1:
            shuffle = True
        momentum = float(input("podaj wartość momentum: "))
        mlp.train(data_for_training, nr_of_epochs, learning_rate, jump, error_level, shuffle, momentum)

    elif output_menu == 6:
        which_data_set = int(input("napisz 1 - zbiór irysów, 2 - autoasocjacja, 3 - własny zbiór: "))
        data_for_testing = None
        if which_data_set == 1:
            data_for_testing = test_data  # trzeba odpowiednio wczytac
        elif which_data_set == 2:
            pass  # TODO dane z drugiego zadania
        else:
            data_for_testing = np.loadtxt(input("podaj nazwę pliku z rozszerzeniem txt: "))


# TODO dodac obsluge momentum
# TODO dodac testowanie sieci