import time
import os
import functions
import torch
import generic_module
from dataset import load_dataset

device = torch.device('cpu')


def get_device() -> torch.device:
    """
    Utility getter for device object
    :return: device object (torch.device)
    """
    return device


def print_time(time_taken: float) -> None:
    """
    Utility function for time printing
    :param time_taken: the time we need to print
    """
    hours, rem = divmod(time_taken, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\tTime taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def main():
    start = time.time()
    load_dataset()
    end = time.time()
    print_time(end - start)

    # 4 neurons per layer
    number_of_neurons = 4

    start = time.time()
    functions.one_hidden_layer_no_activation(number_of_neurons)
    end = time.time()
    print_time(end - start)

    start = time.time()
    functions.two_hidden_layers_sigmoid(number_of_neurons)
    end = time.time()
    print_time(end - start)

    start = time.time()
    functions.two_hidden_layers_relu(number_of_neurons)
    end = time.time()
    print_time(end - start)

    start = time.time()
    functions.two_hidden_layers_relu_SGD_decreasing_lr(number_of_neurons)
    end = time.time()
    print_time(end - start)

    start = time.time()
    functions.two_hidden_layers_relu_adam(number_of_neurons)
    end = time.time()
    print_time(end - start)

    # 32 neurons per layer
    number_of_neurons = 32

    start = time.time()
    functions.one_hidden_layer_no_activation(number_of_neurons)
    end = time.time()
    print_time(end - start)

    start = time.time()
    functions.two_hidden_layers_sigmoid(number_of_neurons)
    end = time.time()
    print_time(end - start)

    start = time.time()
    functions.two_hidden_layers_relu(number_of_neurons)
    end = time.time()
    print_time(end - start)

    start = time.time()
    functions.two_hidden_layers_relu_SGD_decreasing_lr(number_of_neurons)
    end = time.time()
    print_time(end - start)

    start = time.time()
    functions.two_hidden_layers_relu_adam(number_of_neurons)
    end = time.time()
    print_time(end - start)

    start = time.time()
    functions.four_hidden_layers_adam(number_of_neurons)
    end = time.time()
    print_time(end - start)

    start = time.time()
    functions.four_hidden_layers_adam_weight_decay(number_of_neurons)
    end = time.time()
    print_time(end - start)

    start = time.time()
    functions.four_hidden_layers_adam_early_stopping(number_of_neurons)
    end = time.time()
    print_time(end - start)


if __name__ == '__main__':
    # create a directory for saving plots
    generic_module.plot_directory = os.path.join(os.getcwd(), '/images/')
    if not os.path.exists(generic_module.plot_directory):
        os.makedirs(generic_module.plot_directory)

    # update torch device
    if torch.cuda.is_available():
        device = torch.device('cuda')

    print(get_device())
    print('plot images to:', generic_module.plot_directory)

    # start main function
    main()
