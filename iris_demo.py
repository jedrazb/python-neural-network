import numpy as np

from sklearn.metrics import confusion_matrix

from analysis.confustion_matrix import plot_confusion_matrix
from analysis.one_hot_encoder import indices_to_one_hot
from network.network import MultiLayerNetwork
from network.preprocessor import Preprocessor
from network.trainer import Trainer


def main():

    class_labels = ['Virginica', 'Versicolor', 'Setosa']

    dat = np.loadtxt(
        "dataset/iris/iris.data",
        delimiter=',',
    )

    np.random.shuffle(dat)

    # Take first 5 columns as the input X
    x = dat[:, :4]

    # convert the label [0,1,2] representation to one-hot encoding
    y_labels = dat[:, 4:].astype(int)
    y = indices_to_one_hot(y_labels, len(class_labels))

    split_idx = int(0.5 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))

    # Confusion matrix

    cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, class_labels)


if __name__ == "__main__":
    main()
