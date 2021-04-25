import os
import functions
import torch
import generic_module
from dataset import load_dataset, split_training_data_to_validation_set

device = torch.device('cpu')


def get_device() -> torch.device:
    """
    Utility getter for device object
    :return: device object (torch.device)
    """
    return device


def main():
    load_dataset()

    # 4 neurons per layer
    number_of_neurons = 4
    functions.one_hidden_layer_no_activation(number_of_neurons)
    functions.two_hidden_layers_sigmoid(number_of_neurons)
    functions.two_hidden_layers_relu(number_of_neurons)
    functions.two_hidden_layers_relu_SGD_decreasing_lr(number_of_neurons)
    functions.two_hidden_layers_relu_adam(number_of_neurons)

    # 32 neurons per layer
    number_of_neurons = 32
    functions.one_hidden_layer_no_activation(number_of_neurons)
    functions.two_hidden_layers_sigmoid(number_of_neurons)
    functions.two_hidden_layers_relu(number_of_neurons)
    functions.two_hidden_layers_relu_SGD_decreasing_lr(number_of_neurons)
    functions.two_hidden_layers_relu_adam(number_of_neurons)
    functions.four_hidden_layers_adam(number_of_neurons)

    split_training_data_to_validation_set(0.1)

    functions.four_hidden_layers_adam_weight_decay(number_of_neurons)
    functions.four_hidden_layers_adam_early_stopping(number_of_neurons)


if __name__ == '__main__':

    output_dir = os.getcwd()

    # create a directory for saving plots
    generic_module.plot_directory = os.path.join(output_dir, 'images')
    if not os.path.exists(generic_module.plot_directory):
        os.makedirs(generic_module.plot_directory)

    generic_module.last_model_directory = os.path.join(output_dir, 'last_model')
    if not os.path.exists(generic_module.last_model_directory):
        os.makedirs(generic_module.last_model_directory)

    # update torch device
    if torch.cuda.is_available():
        device = torch.device('cuda')

    print(get_device())
    print('plot images to:', generic_module.plot_directory)

    # start main function
    main()
