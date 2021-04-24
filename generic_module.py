import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from assignment_1_code import get_device
from dataset import FashionMNIST_features, get_train_loader, get_validation_loader

plot_directory: str
last_model_directory: str


class GenericFeedforwardNetwork(torch.nn.Module):
    # Define dictionary of non linear activation functions
    non_linear_activation_fun = {'relu': torch.nn.ReLU, 'tanh': torch.nn.Tanh, 'sigmoid': torch.nn.Sigmoid}
    optimizing_func = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}

    def __init__(self, n_features: int, n_hidden_units_per_layer: list, n_outputs: int, activation_fun: str,
                 learning_rate: float, optimizer: str, use_decreasing_learning=False, weight_decay=0,
                 scheduler_gamma=0.1, scheduler_step_size=3):
        """
        Constructor for Generic feed-forward network model
        :param n_features: number of features for input layer
        :param n_hidden_units_per_layer: list for hidden layers
        :param n_outputs: number of output classification options
        :param activation_fun: type of non_linear_activation_fun
        :param learning_rate: float value of the learning rate
        :param use_decreasing_learning: boolean flag for using decreasing_learning with scheduler
        :param weight_decay: the weight decay parameters for optimizer
        :param scheduler_gamma: gamma value for scheduler optimization
        :param scheduler_step_size: step size value for scheduler optimization
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
                    self.non_linear_activation_fun[activation_fun]()
                ]

        # connect hidden layers to fully connected network
        if activation_fun == 'none':
            self.fc_layers = torch.nn.Sequential(*layers)
        else:
            self.fc_layers = torch.nn.Sequential(*layers[:-1])

        # initialize the output layer with log-soft-max activation function
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None

        if optimizer in self.optimizing_func:
            self.optimizer = self.optimizing_func[optimizer](self.parameters(), lr=learning_rate,
                                                             weight_decay=weight_decay)
        if use_decreasing_learning:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=scheduler_step_size, gamma=scheduler_gamma)

        print('\nInitialized Model:')
        print(self)
        print()

    def forward(self, x: object) -> object:
        """
        Function to implement the forward step of the network
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

    def train_model(self, epochs: int, plot_name: str,
                    compute_loss=False, do_early_stopping=False, patience=20) -> None:
        """
        Function to handle network learning step
        :param epochs: number of epochs
        :param plot_name: the name of the plot file to save
        :param compute_loss: boolean flag for loss computing
        :param do_early_stopping: boolean flag for early stopping Regularization
        :param patience: patience to prevent over-fitting in early stopping
        """

        train_acc_per_epoch = []
        val_acc_per_epoch = []
        train_loss_per_epoch = []
        val_loss_per_epoch = []

        # parameters from early stop
        best_early_stop_val_loss_per_epoch = []
        best_early_stop_train_acc_per_epoch = []
        best_early_stop_val_acc_per_epoch = []
        best_early_stop_train_loss_per_epoch = []

        last_acc_improved = -np.inf
        patience_counter = 0

        for i in range(epochs):
            train_loss_for_batch = []
            valid_losses = []

            train_set = get_train_loader()
            valid_set = get_validation_loader()
            for data, label in train_set:
                self.optimizer.zero_grad()

                # flatten the image batch to vector of size 28*28
                data = data.view(-1, FashionMNIST_features).to(get_device())

                # calculate output
                y_prediction = self(data)

                # calculate loss
                loss = self.loss_function(y_prediction, label).to(get_device())

                # backpropagation
                loss.backward()
                self.optimizer.step()
                train_loss_for_batch.append(loss.detach())

            if self.scheduler is not None:
                self.scheduler.step()

            # calculate accuracies and losses
            train_acc = self.calculate_accuracy(train_set)
            val_acc = self.calculate_accuracy(valid_set)

            if do_early_stopping:
                if val_acc < last_acc_improved:
                    patience_counter += 1
                else:
                    # saving the model
                    path_to_model = os.path.join(last_model_directory, 'model.pth')
                    torch.save(self.state_dict(), path_to_model)
                    best_early_stop_val_loss_per_epoch = val_loss_per_epoch.copy()
                    best_early_stop_train_acc_per_epoch = train_acc_per_epoch.copy()
                    best_early_stop_val_acc_per_epoch = val_acc_per_epoch.copy()
                    best_early_stop_train_loss_per_epoch = train_loss_per_epoch.copy()
                    last_acc_improved = val_acc
                    patience_counter = 0

                # stop the learning if the accuracy hasn't improved for the last {patience} iterations
                if patience_counter >= patience:
                    print(f'Early stopping finished after {i} iterations')
                    break

            if compute_loss:
                # check performers on validation set for loss computing
                with torch.no_grad():
                    for data, label in valid_set:
                        # flatten the image to vector of size 28*28
                        data = data.view(-1, FashionMNIST_features).to(get_device())

                        # calculate output and loss for validation
                        y_hat = self(data)
                        loss = self.loss_function(y_hat, label).to(get_device())
                        valid_losses.append(loss.detach())

                # calculate losses for validation set
                valid_losses = np.mean(valid_losses)
                val_loss_per_epoch.append(valid_losses)

                mean_loss_per_batch_in_epoch = np.mean(train_loss_for_batch)
                train_loss_per_epoch.append(mean_loss_per_batch_in_epoch)

                print("epoch {} | train loss : {:.4} | validation loss : {:.4} | train accuracy : {:.4} | validation "
                      "accuracy : {:.4} ".format(i + 1, mean_loss_per_batch_in_epoch, valid_losses, train_acc, val_acc))
            else:
                print("epoch {} | train accuracy : {:.4} | validation "
                      "accuracy : {:.4} ".format(i + 1, train_acc, val_acc))

            train_acc_per_epoch.append(train_acc)
            val_acc_per_epoch.append(val_acc)

        print()
        # load the best model from before early stop
        if do_early_stopping:
            val_loss_per_epoch = best_early_stop_val_loss_per_epoch.copy()
            train_acc_per_epoch = best_early_stop_train_acc_per_epoch.copy()
            val_acc_per_epoch = best_early_stop_val_acc_per_epoch.copy()
            train_loss_per_epoch = best_early_stop_train_loss_per_epoch.copy()

        # plot the results
        acc_plot = plot_name + '_acc.png'
        fig = plt.figure()
        plt.title('Train and validation sets accuracy per epoch')
        plt.plot(train_acc_per_epoch, label='Train Accuracy')
        plt.plot(val_acc_per_epoch, label='Validation Accuracy')
        plt.legend()
        fig.savefig(os.path.join(plot_directory, acc_plot))
        plt.close(fig)
        plt.clf()

        if compute_loss:
            loss_plot = plot_name + '_loss.png'
            fig = plt.figure()
            plt.title('Train and validation sets losses per epoch')
            plt.plot(train_loss_per_epoch, label='Train Loss')
            plt.plot(val_loss_per_epoch, label='Validation Loss')
            plt.legend()
            fig.savefig(os.path.join(plot_directory, loss_plot))
            plt.close(fig)
            plt.clf()

    def calculate_accuracy(self, dataset_loader: DataLoader) -> float:
        """
        Function to calculate the accuracy of a given dataset after model training
        :param dataset_loader: dataset to check of type DataLoader()
        :return: accuracy
        """
        n_correct = 0
        n_total = 0
        for data, label in dataset_loader:
            # flatten the image to vector of size 28*28
            data = data.view(-1, FashionMNIST_features).to(get_device())

            # calculate output
            y_hat = self(data)

            # get the prediction
            predictions = torch.argmax(y_hat, dim=1)
            n_correct += torch.sum(predictions == label.to(get_device())).type(torch.float32)
            n_total += data.shape[0]

        return (n_correct / n_total).item()
