import copy
from dataset import get_test_loader, get_train_loader, get_validation_loader
from generic_module import create_new_network, get_device


def one_hidden_layer_no_activation(number_of_neurons: int) -> None:
    """
    Function 2:
    feed-forward network with one hidden layer.
    no activation functions are applied on the hidden layer (linear).
    The output layer activation function is log softmax.
    :param number_of_neurons: The number of neurons in each hidden layer
    """

    print('\nFunction 2: one_hidden_layer_no_activation')
    learning_rate = 0.01
    model = create_new_network(number_of_neurons, 1, 'none', 'SGD', learning_rate)

    # train the model
    epochs = 50
    model.train_model(epochs, 'Func2', compute_loss=False)

    # print results on train and test sets
    print("\ntrain accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))


def two_hidden_layers_sigmoid(number_of_neurons: int) -> None:
    """
    Function 3:
    Same as Function 2, with 2 hidden layers. Use sigmoid as activation function
    """

    print('\nFunction 3: two_hidden_layers_sigmoid')
    learning_rate = 0.1
    model = create_new_network(number_of_neurons, 2, 'sigmoid', 'SGD', learning_rate)

    # train the model
    epochs = 20
    model.train_model(epochs, 'Func3', compute_loss=False)

    # print results on train and test sets
    print("\ntrain accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))


def two_hidden_layers_relu(number_of_neurons: int) -> None:
    """
    Function 4:
    Same as Function 2, with 2 hidden layers. Use relu as activation function
    Train the network with different learning rates
    """

    print('\nFunction 4: two_hidden_layers_relu')

    # Todo: check start value of learning_rate
    learning_rate = 0.6
    epochs = 20

    # find the best lr with validation set
    best_accuracy = 0
    best_model = None

    while learning_rate < 1:
        # create new model and new optimizer
        model = create_new_network(number_of_neurons, 2, 'relu', 'SGD', learning_rate)

        # train the model with current optimizer
        model.train_model(epochs, f'Func4_lr_{learning_rate}')

        val_acc = model.calculate_accuracy(get_validation_loader())

        print("Learning Rate: {:.4}: Validation Set Accuracy: {:.4}".format(learning_rate, val_acc))

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = copy.deepcopy(model).to(get_device())

        # TODO: check the learning_rate step
        learning_rate += 0.2

    # print results on train and test sets
    print("\ntrain accuracy : %.4f" % best_model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % best_model.calculate_accuracy(get_test_loader()))


def two_hidden_layers_relu_SGD_decreasing_lr(number_of_neurons: int) -> None:
    """
    Function 5:
    Same as Function 4. with 2 hidden layers.
    Train the network with decreasing learning rate
    """

    print('\nFunction 5: two_hidden_layers_relu_SGD_decreasing_lr')

    learning_rate = 0.01
    use_decreasing_learning = True
    model = create_new_network(number_of_neurons, 2, 'relu', 'SGD', learning_rate, use_decreasing_learning)

    # train the model
    epochs = 20
    model.train_model(epochs, 'Func5', compute_loss=False)

    # print results on train and test sets
    print("\ntrain accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))


def two_hidden_layers_relu_adam(number_of_neurons: int) -> None:
    """
    Function 6:
    Same as Function 4. with 2 hidden layers.
    Use Adam as optimizer with lr=0.001
    """

    print('\nFunction 6: two_hidden_layers_relu_adam')

    learning_rate = 0.001
    model = create_new_network(number_of_neurons, 2, 'relu', 'Adam', learning_rate)

    # train the model
    epochs = 30
    model.train_model(epochs, 'Func6', compute_loss=False)

    # print results on train and test sets
    print("\ntrain accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))


def four_hidden_layers_adam(number_of_neurons: int) -> None:
    """
    Function 7:
    Same as Function 6. with 4 hidden layers.
    Use Adam as optimizer with lr=0.001
    """

    print('\nFunction 7: four_hidden_layers_adam')
    learning_rate = 0.001
    model = create_new_network(number_of_neurons, 4, 'relu', 'Adam', learning_rate)

    # train the model
    epochs = 30
    model.train_model(epochs, 'Func7', compute_loss=False)

    # print results on train and test sets
    print("\ntrain accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))


def four_hidden_layers_adam_weight_decay(number_of_neurons):
    """
    Function 8:
    Same as Function 6. with 4 hidden layers.
    Use Adam as optimizer with weight decay as regularization method
    """

    print('\nFunction 8: four_hidden_layers_adam_weight_decay')
    learning_rate = 0.001

    # TODO: check the value of weight_decay
    weight_decay = 0.001
    model = create_new_network(number_of_neurons, 4, 'relu', 'Adam', learning_rate, weight_decay=weight_decay)

    # train the model
    epochs = 250
    model.train_model(epochs, 'Func8', compute_loss=True)

    # print results on train and test sets
    print("\ntrain accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))


def four_hidden_layers_adam_early_stopping(number_of_neurons):
    """
    Function 9:
    Same as Function 6. with 4 hidden layers.
    Use Adam as optimizer and use early stopping regularization (on the validation set) to prevent over-fitting
    """

    print('\nFunction 9: four_hidden_layers_adam_early_stopping')

    learning_rate = 0.001
    model = create_new_network(number_of_neurons, 2, 'relu', 'Adam', learning_rate)

    # train the model
    epochs = 250
    model.train_model(epochs, 'Func9', compute_loss=True, do_early_stopping=True)

    # print results on train and test sets
    print("\ntrain accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))
