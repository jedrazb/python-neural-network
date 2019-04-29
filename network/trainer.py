import numpy as np

from layer.mse_layer import MSELossLayer
from layer.cross_entropy_layer import CrossEntropyLossLayer


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """Constructor.

        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- Training batch size.
            nb_epoch {int} -- Number of training epochs.
            learning_rate {float} -- SGD learning rate to be used in training.
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        if loss_fun == "mse":
            self._loss_layer = MSELossLayer()
        elif loss_fun == "cross_entropy":
            self._loss_layer = CrossEntropyLossLayer()
        else:
            raise ValueError("loss_fun param must be a valid loss function")

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).

        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """

        assert len(input_dataset) == len(target_dataset)
        p = np.random.permutation(len(input_dataset))
        return input_dataset[p], target_dataset[p]

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, ).
        """

        for _ in range(self.nb_epoch):
            if self.shuffle_flag:
                (input_dataset, target_dataset) = self.shuffle(
                    input_dataset, target_dataset)

            epoch_input_dataset = np.array_split(
                input_dataset, self.batch_size)
            epoch_target_dataset = np.array_split(
                target_dataset, self.batch_size)

            for batch_num in range(len(epoch_input_dataset)):
                batch_x = epoch_input_dataset[batch_num]
                batch_y = epoch_target_dataset[batch_num]

                batch_prediction = self.network.forward(batch_x)

                self._loss_layer.forward(batch_prediction, batch_y)
                grad_z = self._loss_layer.backward()

                self.network.backward(grad_z)
                self.network.update_params(self.learning_rate)

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, ).
        """

        predictions = self.network.forward(input_dataset)
        return self._loss_layer.forward(predictions, target_dataset)
