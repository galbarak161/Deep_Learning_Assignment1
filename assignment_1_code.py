import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# global variables
train_loader = None
valid_loader = None
test_loader = None
FashionMNIST_features = 28 * 28
FashionMNIST_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GenericFeedforwardNetwork(torch.nn.Module):
    # Define dictionary of non linear activation functions
    non_linear_activation_fun = {'relu': torch.nn.ReLU, 'tanh': torch.nn.Tanh, 'sigmoid': torch.nn.Sigmoid}

    def __init__(self, n_features, n_hidden_units_per_layer, n_outputs, activation_fun):
        super().__init__()
        dim_list = [n_features, *n_hidden_units_per_layer, n_outputs]
        layers = []
        for in_dim, out_dim in zip(dim_list[:-1], dim_list[1:]):
            if activation_fun == 'none':
                layers += [torch.nn.Linear(in_dim, out_dim)]
            else:
                layers += [
                    torch.nn.Linear(in_dim, out_dim, bias=True),
                    GenericFeedforwardNetwork.non_linear_activation_fun[activation_fun]()
                ]

        if activation_fun == 'none':
            self.fc_layers = torch.nn.Sequential(*layers[:])
        else:
            self.fc_layers = torch.nn.Sequential(*layers[:-1])

        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        h = self.fc_layers(x)
        y_predicted = self.log_softmax(h)
        return y_predicted

    def count_parameters(self) -> int:
        """
        Function to count and return the number of parameters in our network
        :return: number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def calculate_acc(self, dataset_loader: DataLoader) -> float:
        """
        Function to calculate the accuracy of a given dataset after model training
        :param dataset_loader: dataset to check
        :return: accuracy
        """
        global FashionMNIST_features
        n_correct = 0
        n_total = 0
        for j, (data, label) in enumerate(dataset_loader):
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
    The function loads dataset from FashionMNIST
    and prepare the DataLoader objects for training, validation and testing sets
    :return: train_loader, valid_loader, test_loader (update global variables)
    """

    # create normalize MNIST transform
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform_to_tensor = transforms.ToTensor()
    mnist_transforms = transforms.Compose([transform_to_tensor, normalize])

    # load the data: train and test sets
    train_set = torchvision.datasets.FashionMNIST("./data", download=True,
                                                  transform=mnist_transforms)
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False,
                                                 transform=mnist_transforms)
    print('train set len', len(train_set))
    print('test set len', len(test_set))

    # preparing for validation test
    indices = list(range(len(train_set)))
    np.random.shuffle(indices)
    split = int(np.floor(0.8 * len(train_set)))
    train_sample = SubsetRandomSampler(indices[:split])
    valid_sample = SubsetRandomSampler(indices[split:])

    print('train sample len', len(train_sample))
    print('valid sample len', len(valid_sample))

    # create DataLoader object for each set
    global train_loader, valid_loader, test_loader
    train_loader = DataLoader(train_set, sampler=train_sample, batch_size=64)
    valid_loader = DataLoader(train_set, sampler=valid_sample, batch_size=64)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=64, )


def one_hidden_layer_no_activation(number_of_neurons):
    global FashionMNIST_features, FashionMNIST_classes, train_loader, test_loader

    model = GenericFeedforwardNetwork(FashionMNIST_features, [number_of_neurons], FashionMNIST_classes, 'none')

    loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epochs = 50

    for i in range(epochs):
        losses = []
        for j, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()

            # flatten the image to vector of size 28*28
            data = data.view(-1, FashionMNIST_features)

            # calculate output
            y_hat = model(data)

            # calculate loss
            loss = loss_function(y_hat, label)

            # backprop
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        print("epoch {} | train loss : {} ".format(i, np.mean(losses)))

    train_acc = model.calculate_acc(train_loader)
    test_acc = model.calculate_acc(test_loader)
    print("train accuracy : %.4f" % train_acc)
    print("test accuracy : %.4f" % test_acc)


def two_hidden_layers_sigmoid(number_of_neurons):
    pass


def two_hidden_layers_relu(number_of_neurons):
    pass


def two_hidden_layers_relu_SGD_decreasing_lr(number_of_neurons):
    pass


def two_hidden_layers_relu_adam(number_of_neurons):
    pass


def four_hidden_layers_adam(number_of_neurons):
    pass


def four_hidden_layers_adam_weight_decay(number_of_neurons):
    pass


def four_hidden_layers_adam_early_stopping(number_of_neurons):
    pass


if __name__ == '__main__':
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
