from typing import Tuple

import numpy as np
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms

FashionMNIST_features = 28 * 28
FashionMNIST_classes = 10

train_loader_80: DataLoader
valid_loader_20: DataLoader

train_loader_10: DataLoader
valid_loader_90: DataLoader

test_loader: DataLoader
train_set: torchvision.datasets
test_set: torchvision.datasets


def load_dataset():
    """
    Function 1:
    The function loads dataset from FashionMNIST and prepare the DataLoader objects
    :return: None
    """

    print('\nLoad dataset...')

    global train_set, test_set, test_loader, train_loader_80, valid_loader_20, train_loader_10, valid_loader_90

    # create normalize MNIST transform
    # TODO: change to the right numbers
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform_to_tensor = transforms.ToTensor()
    mnist_transforms = transforms.Compose([transform_to_tensor, normalize])

    # load the data: train and test sets
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=mnist_transforms)
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=mnist_transforms)

    print('train set len', len(train_set))
    print('test set len', len(test_set))

    # create DataLoader object for each set
    train_loader_80, valid_loader_20 = split_training_data_to_validation_set(0.8)
    train_loader_10, valid_loader_90 = split_training_data_to_validation_set(0.1)

    test_loader = DataLoader(test_set, shuffle=True, batch_size=64)


def split_training_data_to_validation_set(percent_of_training_set: float):
    """
    Function to split train set into train and validation
    :param percent_of_training_set: percent of training set size (0-1)
    :return: two DataLoader objects - training and validation (Update global variables)
    """

    # preparation
    indices = list(range(len(train_set)))
    np.random.shuffle(indices)
    split = int(np.floor(percent_of_training_set * len(train_set)))
    train_sample = SubsetRandomSampler(indices[:split])
    valid_sample = SubsetRandomSampler(indices[split:])

    print('train sample len', len(train_sample))
    print('valid sample len', len(valid_sample))

    train_loader = DataLoader(train_set, sampler=train_sample, batch_size=64)
    valid_loader = DataLoader(train_set, sampler=valid_sample, batch_size=64)

    return train_loader, valid_loader


def get_train_data(percent_of_training_set: float) -> Tuple[DataLoader, DataLoader]:
    """
    Getter function to train and validation set
    :param percent_of_training_set: percent of training set size (0-1)
    :return: two DataLoader objects - training and validation
    """
    if percent_of_training_set == 0.1:
        return train_loader_80, valid_loader_20

    return train_loader_80, valid_loader_20


def get_test_loader() -> DataLoader:
    return test_loader
