class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """Constructor.

        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension).
            neurons {list} -- Number of neurons in each layer represented as a
                list (the length of the list determines the number of layers).
            activations {list} -- List of the activation function to use for
                each layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        self._layers = []

        n_in = input_dim

        for i in range(len(neurons)):
            n_out = neurons[i]
            act = activations[i]

            self._layers.append(LinearLayer(n_in, n_out))

            if act == 'relu':
                self._layers.append(ReluLayer())
            elif act == 'sigmoid':
                self._layers.append(SigmoidLayer())

            n_in = n_out

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                # _neurons_in_final_layer)
        """

        z = x
        for layer in self._layers:
            z = layer.forward(z)

        return z

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1,
                # _neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """

        z = grad_z
        for layer in self._layers[::-1]:
            z = layer.backward(z)

        return z

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """

        for layer in self._layers:
            layer.update_params(learning_rate)
