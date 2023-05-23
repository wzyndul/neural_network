import numpy as np
import pickle
from project.network import Network


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
        which_data_set = int(input("""
[1] - zbiór irysów
[2] - autoasocjacja
[3] - własny zbiór: """))
        data_for_training = None
        if which_data_set == 1:
            data_for_training = np.genfromtxt('Data/training_data_iris.csv', delimiter=',')
            np.random.shuffle(data_for_training)  # mieszam to bo jest ustawione gatunkami narazie
        elif which_data_set == 2:
            data_for_training = np.genfromtxt('Data/autoasocjacja.csv', delimiter=',')
        else:
            data_for_training = np.loadtxt(input("podaj nazwę pliku z rozszerzeniem txt: "))
        nr_of_epochs = int(input("podaj liczbę epok: "))
        learning_rate = float(input("podaj prędkość nauki: "))
        error_level = float(input("podaj poziom błedu jaki chcesz osiągnąć: "))
        jump = int(input("podaj co ile epok mają być rejestrowane statystyki: "))
        shuffle = False
        if int(input("napisz 1 jeśli chcesz mieszać kolejności prezentacji wzorców: ")) == 1:
            shuffle = True
        momentum = float(input("podaj wartość  współczynnika momentum: "))
        mlp.train(data_for_training, nr_of_epochs, learning_rate, jump, error_level, shuffle, momentum)
        print("statystyki zapisano do pliku!")
        # print(mlp.activations[1])

    elif output_menu == 6:
        which_data_set = int(input("""
[1] - zbiór irysów
[2] - własny zbiór: """))
        data_for_testing = None
        if which_data_set == 1:
            data_for_testing = np.genfromtxt('Data/testing_data_iris.csv', delimiter=',')
            np.random.shuffle(data_for_testing)  # mieszam to bo jest ustawione gatunkami narazie
        elif which_data_set == 2:
            data_for_testing = np.genfromtxt('Data/autoasocjacja.csv', delimiter=',')
        mlp.test(data_for_testing)
        print("statystyki zapisano do pliku!")



# data = []
# with open('Data/iris.csv', 'r') as file:
#     for line in file:
#         line = line.strip()
#         if line:
#             values = line.split(',')
#             features = [float(value) for value in values[:4]]
#             if values[4] == 'Iris-setosa':
#                 target = [1, 0, 0]
#             if values[4] == 'Iris-versicolor':
#                 target = [0, 1, 0]
#             if values[4] == 'Iris-virginica':
#                 target = [0, 0, 1]
#             data.append(features + target)
#
# data = np.array(data)
#
# # mam po 50 przypadków z każdego
# data_setosa = data[:50, :]
# data_versicolor = data[50:100, :]
# data_virginica = data[100:150, :]
#
# #mieszam je
# np.random.shuffle(data_setosa)
# np.random.shuffle(data_versicolor)
# np.random.shuffle(data_virginica)
#
# selected_samples_setosa = data_setosa[:10, :]
# selected_samples_vers = data_versicolor[:10, :]
# selected_samples_virg = data_virginica[:10, :]
#
# testing_data = np.concatenate((selected_samples_setosa, selected_samples_vers, selected_samples_virg), axis=0)
#
# np.savetxt('testing_data_iris.csv', testing_data, delimiter=',', fmt='%.1f')
#
# remaining_samples_setosa = data_setosa[10:, :]
# remaining_samples_vers = data_versicolor[10:, :]
# remaining_samples_virgi = data_virginica[10:, :]
#
# training_data = np.concatenate((remaining_samples_setosa, remaining_samples_vers, remaining_samples_virgi), axis=0)
#
# np.savetxt('training_data_iris.csv', training_data, delimiter=',', fmt='%.1f')