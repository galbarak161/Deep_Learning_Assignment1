import copy
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# global variables
train_loader = []
valid_loader = []
test_loader = []
train_set = []
FashionMNIST_features = 28 * 28
FashionMNIST_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GenericFeedforwardNetwork(torch.nn.Module):
    # Define dictionary of non linear activation functions
    non_linear_activation_fun = {'relu': torch.nn.ReLU, 'tanh': torch.nn.Tanh, 'sigmoid': torch.nn.Sigmoid}

    def __init__(self, n_features: int, n_hidden_units_per_layer: list, n_outputs: int, activation_fun: str):
        """
        Constructor for Generic feed-forward network model
        :param n_features: number of features for input layer
        :param n_hidden_units_per_layer: list for hidden layers
        :param n_outputs: number of output classification options
        :param activation_fun: type of non_linear_activation_fun
        :return: new model instance
        """
        super().__init__()
        dim_list = [n_features, *n_hidden_units_per_layer, n_outputs]
        layers = []
        # create the hidden layers
        for in_dim, out_dim in zip(dim_list[:-1], dim_list[1:]):
            if activation_fun == 'none':
                layers += [torch.nn.Linear(in_dim, out_dim)]
            else:
                layers += [
                    torch.nn.Linear(in_dim, out_dim, bias=True),
                    GenericFeedforwardNetwork.non_linear_activation_fun[activation_fun]()
                ]

        # connect hidden layers to fully connected network
        if activation_fun == 'none':
            self.fc_layers = torch.nn.Sequential(*layers[:])
        else:
            self.fc_layers = torch.nn.Sequential(*layers[:-1])

        # initialize the output layer with log-soft-max activation function
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x: object) -> object:
        """
        Function for implement single forward step on network
        :param x: inputs (tensor)
        :return: prediction according to tensor inputs (tensor)
        """
        h = self.fc_layers(x)
        y_predicted = self.log_softmax(h)
        return y_predicted

    def count_parameters(self) -> int:
        """
        Function to count and return the number of parameters in our network
        :return: number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train_model(self, optimizer: torch, epochs: int, loss_function, use_entire_training_set=True) -> None:
        """
        Function to handle network learning step
        :param optimizer: torch optimizer after initialization
        :param epochs: number of epochs
        :param loss_function: torch loss function after initialization
        :param use_entire_training_set: bolean flag for using entire dataset
        """
        global train_loader, valid_loader

        # for debug
        epochs = 2

        if use_entire_training_set:
            training_set = train_loader
            valid_set = valid_loader
        else:
            training_set, valid_set = split_training_data_to_validation_set(0.1)

        train_acc_per_epoch = []
        val_acc_per_epoch = []
        train_loss_per_epoch = []
        val_loss_per_epoch = []

        for i in range(epochs):
            train_losses = []
            valid_losses = []
            for data, label in training_set:
                optimizer.zero_grad()

                # flatten the image batch to vector of size 28*28
                data = data.view(-1, FashionMNIST_features)

                # calculate output
                y_prediction = self(data)

                # calculate loss
                loss = loss_function(y_prediction, label)

                # backpropagation
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach())

            # check performers on validation set
            for data, label in valid_set:
                # flatten the image to vector of size 28*28
                data = data.view(-1, FashionMNIST_features)

                # calculate output and loss for validation
                y_hat = self(data)
                loss = loss_function(y_hat, label)
                valid_losses.append(loss.detach())

            # calculate accuracies and losses
            train_acc = self.calculate_accuracy(training_set)
            val_acc = self.calculate_accuracy(valid_set)

            train_losses = np.mean(train_losses)
            valid_losses = np.mean(valid_losses)
            print("epoch {} | train loss : {} | validation loss : {} | train accuracy : {} | validation accuracy : {} ".
                  format(i + 1, train_losses, valid_losses, train_acc, val_acc))

            train_acc_per_epoch.append(train_acc)
            val_acc_per_epoch.append(val_acc)
            train_loss_per_epoch.append(train_losses)
            val_loss_per_epoch.append(valid_losses)

        # plot the results
        plt.title('Train and validation sets accuracy per epoch')
        plt.plot(train_acc_per_epoch, label='Train Accuracy')
        plt.plot(val_acc_per_epoch, label='Validation Accuracy')
        plt.xticks(range(epochs))
        plt.legend()
        plt.show()

        plt.title('Train and validation sets losses per epoch')
        plt.plot(train_loss_per_epoch, label='Train Loss')
        plt.plot(val_loss_per_epoch, label='Validation Loss')
        plt.xticks(range(epochs))
        plt.legend()
        plt.show()

    def calculate_accuracy(self, dataset_loader) -> float:
        """
        Function to calculate the accuracy of a given dataset after model training
        :param dataset_loader: dataset to check of type DataLoader()
        :return: accuracy
        """
        global FashionMNIST_features

        n_correct = 0
        n_total = 0
        for data, label in dataset_loader:
            # flatten the image to vector of size 28*28
            data = data.view(-1, FashionMNIST_features)

            # calculate output
            y_hat = self(data)

            # get the prediction
            predictions = torch.argmax(y_hat, dim=1)
            n_correct += torch.sum(predictions == label).type(torch.float32)
            n_total += data.shape[0]

        acc = (n_correct / n_total).item()
        return acc


def load_dataset() -> None:
    """
    Function 1:
    The function loads dataset from FashionMNIST and prepare the DataLoader objects
    :return: train_loader, valid_loader, test_loader (update global variables)
    """

    # create normalize MNIST transform
    # TODO: change to the right numbers
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform_to_tensor = transforms.ToTensor()
    mnist_transforms = transforms.Compose([transform_to_tensor, normalize])

    # load the data: train and test sets
    global train_set
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=mnist_transforms)
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=mnist_transforms)
    print('train set len', len(train_set))
    print('test set len', len(test_set))

    # create DataLoader object for each set
    global train_loader, valid_loader, test_loader
    train_loader, valid_loader = split_training_data_to_validation_set(0.8)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=64)


def split_training_data_to_validation_set(percent_of_training_set: float) -> DataLoader:
    """
    Function for implement single forward step on network
    :param percent_of_training_set: percent of training set size (0-1)
    :return: two DataLoader objects - training and validation
    """

    global train_set
    # preparation
    indices = list(range(len(train_set)))
    np.random.shuffle(indices)
    split = int(np.floor(percent_of_training_set * len(train_set)))
    train_sample = SubsetRandomSampler(indices[:split])
    valid_sample = SubsetRandomSampler(indices[split:])

    print('train sample len', len(train_sample))
    print('valid sample len', len(valid_sample))

    train = DataLoader(train_set, sampler=train_sample, batch_size=64)
    valid = DataLoader(train_set, sampler=valid_sample, batch_size=64)

    return train, valid


def one_hidden_layer_no_activation(number_of_neurons: int) -> None:
    """
    Function 2:
    feed-forward network with one hidden layer.
    no activation functions are applied on the hidden layer (linear).
    The output layer activation function is log softmax.
    :param number_of_neurons: The number of neurons in each hidden layer
    """

    global FashionMNIST_features, FashionMNIST_classes, train_loader, test_loader

    print('\nRunning one_hidden_layer_no_activation')
    model = GenericFeedforwardNetwork(FashionMNIST_features, [number_of_neurons], FashionMNIST_classes, 'none')

    loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epochs = 50

    # train the model
    model.train_model(optimizer, epochs, loss_function)

    # print results on train and test sets
    print("train accuracy : %.4f" % model.calculate_accuracy(train_loader))
    print("test accuracy : %.4f" % model.calculate_accuracy(test_loader))


def two_hidden_layers_sigmoid(number_of_neurons: int) -> None:
    """
    Function 3:
    Same as Function 2, with 2 hidden layers. Use sigmoid as activation function
    """

    global FashionMNIST_features, FashionMNIST_classes, train_loader, test_loader

    print('\nRunning two_hidden_layers_sigmoid')
    model = GenericFeedforwardNetwork(FashionMNIST_features, [number_of_neurons] * 2, FashionMNIST_classes, 'sigmoid')

    loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epochs = 20

    # train the model
    model.train_model(optimizer, epochs, loss_function)

    # print results on train and test sets
    print("train accuracy : %.4f" % model.calculate_accuracy(train_loader))
    print("test accuracy : %.4f" % model.calculate_accuracy(test_loader))


def two_hidden_layers_relu(number_of_neurons: int) -> None:
    """
    Function 4:
    Same as Function 2, with 2 hidden layers. Use relu as activation function
    Train the network with different learning rates
    """

    global FashionMNIST_features, FashionMNIST_classes, train_loader, test_loader, valid_loader

    print('\nRunning two_hidden_layers_relu')

    loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = 0.01
    epochs = 20

    # find the best lr with validation set
    best_accuracy = 0
    best_model = None

    while learning_rate < 1:
        # create new model and new optimizer
        model = GenericFeedforwardNetwork(FashionMNIST_features, [number_of_neurons] * 2, FashionMNIST_classes, 'relu')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # train the model with current optimizer
        model.train_model(optimizer, epochs, loss_function)

        val_acc = model.calculate_accuracy(valid_loader)
        print(f'Learning Rate: {learning_rate}: Validation Set Accuracy: {val_acc}')
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = copy.deepcopy(model)

        learning_rate += 0.3

    train_acc = best_model.calculate_accuracy(train_loader)
    test_acc = best_model.calculate_accuracy(test_loader)
    print("train accuracy : %.4f" % train_acc)
    print("test accuracy : %.4f" % test_acc)


def two_hidden_layers_relu_SGD_decreasing_lr(number_of_neurons: int) -> None:
    """
    Function 5:
    Same as Function 4. with 2 hidden layers.
    Train the network with decreasing learning rate
    """

    global FashionMNIST_features, FashionMNIST_classes, train_loader, test_loader, valid_loader

    print('\nRunning two_hidden_layers_relu_SGD_decreasing_lr')

    loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = 0.01
    epochs = 20

    # find the best lr with validation set
    best_accuracy = 0
    best_model = None

    while learning_rate > 0:
        # create new model and new optimizer
        model = GenericFeedforwardNetwork(FashionMNIST_features, [number_of_neurons] * 2, FashionMNIST_classes, 'relu')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # train the model with current optimizer
        model.train_model(optimizer, epochs, loss_function)

        val_acc = model.calculate_accuracy(valid_loader)
        print(f'Learning Rate: {learning_rate}: Validation Set Accuracy: {val_acc}')
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = copy.deepcopy(model)

        learning_rate -= 0.001

    train_acc = best_model.calculate_accuracy(train_loader)
    test_acc = best_model.calculate_accuracy(test_loader)
    print("train accuracy : %.4f" % train_acc)
    print("test accuracy : %.4f" % test_acc)


def two_hidden_layers_relu_adam(number_of_neurons: int) -> None:
    """
    Function 6:
    Same as Function 4. with 2 hidden layers.
    Use Adam as optimizer with lr=0.001
    """

    global FashionMNIST_features, FashionMNIST_classes, train_loader, test_loader

    print('\nRunning two_hidden_layers_relu_adam')
    model = GenericFeedforwardNetwork(FashionMNIST_features, [number_of_neurons] * 2, FashionMNIST_classes, 'relu')

    loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 30

    # train the model
    model.train_model(optimizer, epochs, loss_function)

    # print results on train and test sets
    print("train accuracy : %.4f" % model.calculate_accuracy(train_loader))
    print("test accuracy : %.4f" % model.calculate_accuracy(test_loader))


def four_hidden_layers_adam(number_of_neurons: int) -> None:
    """
    Function 7:
    Same as Function 6. with 4 hidden layers.
    Use Adam as optimizer with lr=0.001 with only 10% of the training set for training
    """

    global FashionMNIST_features, FashionMNIST_classes, train_loader, test_loader

    print('\nRunning four_hidden_layers_adam')
    model = GenericFeedforwardNetwork(FashionMNIST_features, [number_of_neurons] * 4, FashionMNIST_classes, 'relu')

    loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 250

    # train the model
    model.train_model(optimizer, epochs, loss_function, False)

    # print results on train and test sets
    print("train accuracy : %.4f" % model.calculate_accuracy(train_loader))
    print("test accuracy : %.4f" % model.calculate_accuracy(test_loader))


def four_hidden_layers_adam_weight_decay(number_of_neurons):
    pass


def four_hidden_layers_adam_early_stopping(number_of_neurons):
    pass


def main():
    load_dataset()

    # 4 neurons per layer
    number_of_neurons = 4
    one_hidden_layer_no_activation(number_of_neurons)
    two_hidden_layers_sigmoid(number_of_neurons)
    two_hidden_layers_relu(number_of_neurons)
    two_hidden_layers_relu_SGD_decreasing_lr(number_of_neurons)
    two_hidden_layers_relu_adam(number_of_neurons)

    # 32 neurons per layer
    number_of_neurons = 32
    one_hidden_layer_no_activation(number_of_neurons)
    two_hidden_layers_sigmoid(number_of_neurons)
    two_hidden_layers_relu(number_of_neurons)
    two_hidden_layers_relu_SGD_decreasing_lr(number_of_neurons)
    two_hidden_layers_relu_adam(number_of_neurons)
    four_hidden_layers_adam(number_of_neurons)
    four_hidden_layers_adam_weight_decay(number_of_neurons)
    four_hidden_layers_adam_early_stopping(number_of_neurons)


if __name__ == '__main__':
    main()
