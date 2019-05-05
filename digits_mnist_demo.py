import numpy as np
import gzip

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from analysis.confustion_matrix import plot_confusion_matrix
from analysis.one_hot_encoder import indices_to_one_hot
from network.network import MultiLayerNetwork
from network.preprocessor import Preprocessor
from network.trainer import Trainer


filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

prefix_path = "dataset/digits_mnist"

file_path = "{prefix}/{file}"


def main():

    class_labels = [str(x) for x in range(10)]

    train_x, train_labels, test_x, test_labels = load_mnist()

    # Convert the label class into a one-hot representation
    train_y = indices_to_one_hot(train_labels, 10)
    test_y = indices_to_one_hot(test_labels, 10)

    train_x = train_x / 255
    test_x = test_x / 255

    input_dim = 784
    neurons = [128, 128, 10]
    activations = ["relu", "relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=100,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(train_x, train_y)
    print("Train loss = ", trainer.eval_loss(train_x, train_y))
    print("Validation loss = ", trainer.eval_loss(test_x, test_y))

    preds = net(test_x).argmax(axis=1).squeeze()
    targets = test_y.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))

    # Confusion matrix

    cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, class_labels)


def load_mnist():
    mnist = {}
    for name in filename[:2]:
        path = file_path.format(prefix=prefix_path, file=name[1])
        with gzip.open(path, 'rb') as f:
            mnist[name[0]] = np.frombuffer(
                f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    for name in filename[-2:]:
        path = file_path.format(prefix=prefix_path, file=name[1])
        with gzip.open(path, 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def visualise_image(label, x_set, y_set):
    img_idx = np.where(y_set == label)[0][0]
    img = np.reshape(x_set[img_idx], (28, 28))
    plt.figure()
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
