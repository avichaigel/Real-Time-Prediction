# first neural network with keras tutorial
import sys

from numpy import loadtxt
from sklearn.metrics import fbeta_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler



def f025(y_true, y_pred):
    return fbeta_score(y_true, y_pred, average='binary', beta=0.25)


def create_model():
    model = Sequential()
    model.add(Dense(120, input_dim=120, activation='relu'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(70, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_test_indexes():
    list = []
    for i in range(121):
        list.append(i)
    list.pop(0)
    return list


def main():
    # load the dataset
    train = loadtxt('train.csv', delimiter=',')
    validate = loadtxt('validate.csv', delimiter=',')
    test_indexes = get_test_indexes()
    test = loadtxt('test.csv', delimiter=',', usecols = (test_indexes))
    standard_scaler = StandardScaler()

    # split into input (X) and output (y) variables
    X_train = train[0:300000, 1:121]
    X_train = standard_scaler.fit_transform(X_train)
    y_train = train[0:300000, 0]
    X_mutaion1 = train[300000:400000, 1:121]
    X_mutaion1 = standard_scaler.fit_transform(X_mutaion1)
    y_mutaion1 = train[300000:400000, 0]
    X_mutaion2 = train[400000:500000, 1:121]
    X_mutaion2 = standard_scaler.fit_transform(X_mutaion2)
    y_mutaion2 = train[400000:500000, 0]
    X_mutaion3 = train[500000:600000, 1:121]
    X_mutaion3 = standard_scaler.fit_transform(X_mutaion3)
    y_mutaion3 = train[500000:600000, 0]
    X_mutaion4 = train[600000:700000, 1:121]
    X_mutaion4 = standard_scaler.fit_transform(X_mutaion4)
    y_mutaion4 = train[600000:700000, 0]

    X_validate = validate[:, 1:121]
    X_validate = standard_scaler.fit_transform(X_validate)
    y_validate = validate[:, 0]
    X_test = test[:, 0:120]
    X_test = standard_scaler.fit_transform(X_test)

    # create the keras models
    model1 = create_model()
    model2 = create_model()
    model3 = create_model()
    model4 = create_model()
    model5 = create_model()

    population = [model1, model2, model3, model4, model5]

    # compile the keras models
    for model in population:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    results_dict = {}
    file_index = 0
    for epoch in range(7):
        best_counter = 0
        f025_res_max = 0
        for model in population:
            predictions = model.predict_classes(X_train)
            f025_res = f025(y_train, predictions)
            if f025_res > f025_res_max:
                f025_res_max = f025_res
                max_f025_index = best_counter
            best_counter += 1

        # max_f025_index is the index of best in population
        best = population[max_f025_index]
        for i in range(5):
            del population[0]
        population.append(best)
        # now only best is in population

        predictions = best.predict_classes(X_validate)
        f025_res = f025(y_validate, predictions)

        _, accuracy = best.evaluate(X_validate, y_validate)

        print("")
        print("----------------------------------------")
        print("epoch number: ", epoch + 1)
        print("best result on validate: ", f025_res)
        print('Accuracy: ', accuracy)
        print("----------------------------------------")
        print("")

        # save best to file & insert (file, res)->dict
        best.save("best.h5")
        file_for_dict_name = "model" + str(file_index)
        best.save(file_for_dict_name)
        results_dict[file_for_dict_name] = f025_res

        mutation1 = load_model('best.h5')
        mutation1.fit(X_mutaion1, y_mutaion1, epochs=30, batch_size=32768)
        population.append(mutation1)

        mutation2 = load_model('best.h5')
        mutation2.fit(X_mutaion2, y_mutaion2, epochs=30, batch_size=32768)
        population.append(mutation2)

        mutation3 = load_model('best.h5')
        mutation3.fit(X_mutaion3, y_mutaion3, epochs=30, batch_size=32768)
        population.append(mutation3)

        mutation4 = load_model('best.h5')
        mutation4.fit(X_mutaion4, y_mutaion4, epochs=30, batch_size=32768)
        population.append(mutation4)

        file_index += 1

    # best on test
    best_file = max(results_dict, key=results_dict.get)
    best = load_model(best_file)
    predictions = best.predict_classes(X_test)
    predictions.tofile('316096338_308178136_2.txt', sep="\n", format="%s")




if __name__ == '__main__':
    main()

