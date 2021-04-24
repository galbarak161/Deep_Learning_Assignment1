from dataset import FashionMNIST_classes, FashionMNIST_features
from generic_module import GenericFeedforwardNetwork
from assignment_1_code import get_device


class MnistFashionFeedforwardNetwork(GenericFeedforwardNetwork):

    def __init__(self, n_hidden_units_per_layer: list, activation_fun: str, learning_rate: float, optimizer: str,
                 use_decreasing_learning=False, weight_decay=0, scheduler_gamma=0.1, scheduler_step_size=3):

        super().__init__(FashionMNIST_features, n_hidden_units_per_layer, FashionMNIST_classes, activation_fun,
                         learning_rate, optimizer, use_decreasing_learning, weight_decay,
                         scheduler_gamma, scheduler_step_size)

        self.to(get_device())
