import copy
import os

import torch
import generic_module

from assignment_1_code import get_device
from dataset import get_test_loader, get_train_loader, get_validation_loader
from mnist_fashion_module import MnistFashionFeedforwardNetwork


def one_hidden_layer_no_activation(number_of_neurons: int) -> None:
    """
    Function 2:
    feed-forward network with one hidden layer.
    no activation functions are applied on the hidden layer (linear).
    The output layer activation function is log softmax.
    :param number_of_neurons: The number of neurons in each hidden layer
    """

    print('Function 2: one_hidden_layer_no_activation')

    model = MnistFashionFeedforwardNetwork(
        n_hidden_units_per_layer=[number_of_neurons],
        activation_fun='none',
        learning_rate=0.01,
        optimizer='SGD',
    )

    # train the model
    epochs = 50
    model.train_model(epochs, f'{number_of_neurons} - Func2')

    # print results on train and test sets
    print("train accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))


def two_hidden_layers_sigmoid(number_of_neurons: int) -> None:
    """
    Function 3:
    Same as Function 2, with 2 hidden layers. Use sigmoid as activation function
    """

    print('Function 3: two_hidden_layers_sigmoid')

    model = MnistFashionFeedforwardNetwork(
        n_hidden_units_per_layer=[number_of_neurons] * 2,
        activation_fun='sigmoid',
        learning_rate=0.1,
        optimizer='SGD',
    )

    # train the model
    epochs = 20
    model.train_model(epochs, f'{number_of_neurons} - Func3')

    # print results on train and test sets
    print("train accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))


def two_hidden_layers_relu(number_of_neurons: int) -> None:
    """
    Function 4:
    Same as Function 2, with 2 hidden layers. Use relu as activation function
    Train the network with different learning rates
    """

    print('Function 4: two_hidden_layers_relu')

    learning_rate = 0.01
    step_size = 0.01
    learning_rate_stop_value = 0.15
    epochs = 20

    # find the best lr with validation set
    best_accuracy = 0
    best_model = None
    best_learning_rate = learning_rate

    while learning_rate < learning_rate_stop_value:
        # create new model and new optimizer
        model = MnistFashionFeedforwardNetwork(
            n_hidden_units_per_layer=[number_of_neurons] * 2,
            activation_fun='relu',
            learning_rate=learning_rate,
            optimizer='SGD',
        )

        # train the model with current optimizer
        model.train_model(epochs, f'{number_of_neurons} - Func4_lr_{learning_rate}')

        val_acc = model.calculate_accuracy(get_validation_loader())

        print("Learning Rate: {:.4}: Validation Set Accuracy: {:.4}".format(learning_rate, val_acc))

        if val_acc > best_accuracy:
            best_learning_rate = learning_rate
            best_accuracy = val_acc
            best_model = copy.deepcopy(model).to(get_device())

        learning_rate += step_size

    # print results on train and test sets
    print(f'best learning rate is {best_learning_rate}')
    print("train accuracy : %.4f" % best_model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % best_model.calculate_accuracy(get_test_loader()))


def two_hidden_layers_relu_SGD_decreasing_lr(number_of_neurons: int) -> None:
    """
    Function 5:
    Same as Function 4. with 2 hidden layers.
    Train the network with decreasing learning rate
    """

    print('Function 5: two_hidden_layers_relu_SGD_decreasing_lr')

    gamma = 0.8
    step_size = 3
    if number_of_neurons == 32:
        step_size = 5

    model = MnistFashionFeedforwardNetwork(
        n_hidden_units_per_layer=[number_of_neurons] * 2,
        activation_fun='relu',
        learning_rate=0.01,
        optimizer='SGD',
        use_decreasing_learning=True,
        scheduler_gamma=gamma,
        scheduler_step_size=step_size
    )

    # train the model
    epochs = 20
    model.train_model(epochs, f'{number_of_neurons} - Func5_step_{step_size}')

    # print results on train and test sets
    print("train accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))


def two_hidden_layers_relu_adam(number_of_neurons: int) -> None:
    """
    Function 6:
    Same as Function 4. with 2 hidden layers.
    Use Adam as optimizer with lr=0.001
    """

    print('Function 6: two_hidden_layers_relu_adam')

    model = MnistFashionFeedforwardNetwork(
        n_hidden_units_per_layer=[number_of_neurons] * 2,
        activation_fun='relu',
        learning_rate=0.001,
        optimizer='Adam',
    )

    # train the model
    epochs = 30
    model.train_model(epochs, f'{number_of_neurons} - Func6')

    # print results on train and test sets
    print("train accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))


def four_hidden_layers_adam(number_of_neurons: int) -> None:
    """
    Function 7:
    Same as Function 6. with 4 hidden layers.
    Use Adam as optimizer with lr=0.001
    """

    print('Function 7: four_hidden_layers_adam')

    model = MnistFashionFeedforwardNetwork(
        n_hidden_units_per_layer=[number_of_neurons] * 4,
        activation_fun='relu',
        learning_rate=0.001,
        optimizer='Adam',
    )

    # train the model
    epochs = 30
    model.train_model(epochs, f'{number_of_neurons} - Func7', compute_loss=True)

    # print results on train and test sets
    print("train accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))


def four_hidden_layers_adam_weight_decay(number_of_neurons):
    """
    Function 8:
    Same as Function 6. with 4 hidden layers.
    Use Adam as optimizer with weight decay as regularization method
    """

    print('Function 8: four_hidden_layers_adam_weight_decay')

    weight_decay = 0.001
    model = MnistFashionFeedforwardNetwork(
        n_hidden_units_per_layer=[number_of_neurons] * 4,
        activation_fun='relu',
        learning_rate=0.001,
        optimizer='Adam',
        weight_decay=weight_decay
    )

    # train the model
    epochs = 250
    model.train_model(epochs, f'{number_of_neurons} - Func8', compute_loss=True)

    # print results on train and test sets
    print("train accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))


def four_hidden_layers_adam_early_stopping(number_of_neurons):
    """
    Function 9:
    Same as Function 6. with 4 hidden layers.
    Use Adam as optimizer and use early stopping regularization (on the validation set) to prevent over-fitting
    """

    print('Function 9: four_hidden_layers_adam_early_stopping')

    model = MnistFashionFeedforwardNetwork(
        n_hidden_units_per_layer=[number_of_neurons] * 4,
        activation_fun='relu',
        learning_rate=0.001,
        optimizer='Adam',
    )

    # train the model
    epochs = 250
    path_to_model = os.path.join(generic_module.last_model_directory, 'model.pth')
    model.train_model(epochs, f'{number_of_neurons} - Func9', compute_loss=True, do_early_stopping=True)
    model.load_state_dict(torch.load(path_to_model))

    # print results on train and test sets
    print("train accuracy : %.4f" % model.calculate_accuracy(get_train_loader()))
    print("test accuracy : %.4f" % model.calculate_accuracy(get_test_loader()))
