# MLP neural network implemented using numpy

This is a simple implementation of multilayer perceptron (feedforward) neural network. The netowrk is implemented using numpy in python.

## General 

This a fully functional neural network library with a demo (iris dataset). The implemented features are:
- loss functions: cross entropy, mean squared error
- layers: linear, sigmoid, ReLU
- network with forward and backpropagation
- function to one hot encode labels
- confussion matrix visualiser

There are lots of comments in the code explaining the details

## Rquirements

- python 3.x
- numpy

## Installation 

To install required dependencies: `make install`. To run the demo: `python3 iris_demo.py`.

## Demo (iris dataset)

The demo is based on the [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), the dataset can be found in `dataset/iris.data`.

In the demo data is shuffled split into train and validation datasets, the netowork is trained and the performance is displayed as a confusion matrix:

<p align="center"><img src="https://i.ibb.co/kGJL9vf/Screenshot-2019-05-04-at-18-52-18.png" width="400"></p>
