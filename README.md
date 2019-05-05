# MLP neural network library implemented using numpy



## General 

This a fully functional feedforward neural network library. The implemented features are:
- loss functions: cross entropy, mean squared error
- layers: linear, sigmoid, ReLU
- network with forward and backpropagation
- function to one hot encode labels
- confussion matrix visualiser

There are two demos to demonstrate capabilities of the library:
- iris dataset classifier
- handwritten digits (mnist) classifier

There are lots of comments in the code explaining the details

## Rquirements

- python 3.x
- numpy
- matplotlib

## Installation 

To install required dependencies: `make install`.

## Demo (iris dataset)

To run the demo: `python3 iris_demo.py`.

The demo is based on the [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), the dataset can be found in `dataset/iris/`. It consists of 150 entries. 

The accuracy obtained on the validation set is: 98.7%.

In the demo data is shuffled split into train and validation datasets, the netowork is trained and the performance is displayed as a confusion matrix:

<p align="center"><img src="https://i.ibb.co/RBBWTTJ/Screenshot-2019-05-05-at-12-50-43.png" width="400"></p>

## Demo (handwritten digits mnist dataset)

To run the demo: `python3 digits_mnist_demo.py`.

This demo is based on [mnist digits dataset](http://yann.lecun.com/exdb/mnist/), the dataset can be found in `dataset/digits_mnist/`. It consists of 60000 train images and 10000 test images.

The accuracy obtained on the test set is: 97.8%.

Example network architecture:

<p align="center"><img src="https://www.newtownpartners.com/wp-content/uploads/2018/08/jon-jon-4.png" width="400"></p>

Example confusion matrix:

<p align="center"><img src="https://i.ibb.co/fvdh3Kt/Screenshot-2019-05-05-at-12-35-39.png" width="400"></p>





