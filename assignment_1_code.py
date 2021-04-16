import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
import matplotlib.pyplot as plt



def load_dataset():
    # load the data: train and test sets
    train_set = torchvision.datasets.FashionMNIST("./data", download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False,
                                                 transform=transforms.Compose([transforms.ToTensor()]))
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

    # create Data Loader object
    train_loader = DataLoader(train_set, sampler=train_sample, batch_size=64)
    valid_loader = DataLoader(train_set, sampler=train_sample, batch_size=64)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=64, )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

load_dataset()
