import numpy as np
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms

FashionMNIST_features = 28 * 28
FashionMNIST_classes = 10

train_loader: DataLoader
valid_loader: DataLoader
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

    global train_set, test_set, test_loader

    # create normalize MNIST transform
    data_mean = 0.2860405969887955
    data_std = 0.35302424451492237
    normalize = transforms.Normalize((data_mean,), (data_std,))
    transform_to_tensor = transforms.ToTensor()
    mnist_transforms = transforms.Compose([transform_to_tensor, normalize])

    # load the data: train and test sets
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=mnist_transforms)
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=mnist_transforms)

    print('train set len', len(train_set))
    print('test set len', len(test_set))

    # create DataLoader object for each set
    split_training_data_to_validation_set(0.8)
    test_loader = DataLoader(test_set, batch_size=64)


def split_training_data_to_validation_set(percent_of_training_set: float) -> None:
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

    print(f'split training data to {percent_of_training_set} train and {1 - percent_of_training_set} validation')
    print('train sample len', len(train_sample))
    print('valid sample len', len(valid_sample))

    global train_loader, valid_loader
    train_loader = DataLoader(train_set, sampler=train_sample, batch_size=64)
    valid_loader = DataLoader(train_set, sampler=valid_sample, batch_size=64)


def get_train_loader() -> DataLoader:
    """
    getter function for train_loader
    """
    return train_loader


def get_validation_loader() -> DataLoader:
    """
    getter function for valid_loader
    """
    return valid_loader


def get_test_loader() -> DataLoader:
    """
    getter function for test_loader
    """
    return test_loader
