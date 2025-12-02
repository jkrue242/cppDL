//==============================================
// joseph krueger, 2025
//==============================================
#include "network/network.hpp"
#include "network/functions.hpp"
#include "network/layer.hpp"
#include "network/loss_layer.hpp"
#include "network/softmax_cross_entropy_loss.hpp"
#include "train.hpp"
#include "Eigen/Dense"
#include <iostream>
#include <random>
#include <memory>
#include <vector>

//==============================================
// This is an example of training a network to 
// learn the "XOR" function
//==============================================

// random number generator 
std::random_device rd;
std::mt19937 rand_engine(rd());
std::uniform_real_distribution<float> distr(-1, 1);

//==============================================
// driver
//==============================================
int main() {

    // XOR dataset
    const int features = 2;
    Eigen::MatrixXf X_train(features, 4);
    X_train << 0.0, 1.0, 0.0, 1.0, 
               0.0, 0.0, 1.0, 1.0; 

    // one hot encoded
    const int classes = 2;
    Eigen::MatrixXf Y_train(classes, 4);
    Y_train << 1.0, 0.0, 0.0, 1.0, 
                0.0, 1.0, 1.0, 0.0;

    int num_samples = X_train.cols();

    // build the network
    std::vector<std::unique_ptr<Layer>> layers;
    layers.push_back(std::make_unique<Layer>(features, 16, "ReLU"));
    layers.push_back(std::make_unique<Layer>(16, 16, "Sigmoid"));
    layers.push_back(std::make_unique<Layer>(16, classes, "Identity")); // last layer must have the identity activation (i.e., no activation)
    std::unique_ptr<LossLayer> loss_layer = std::make_unique<SoftmaxCrossEntropyLoss>();
    Network network(std::move(layers), std::move(loss_layer));

    // train the network
    const float learning_rate = 0.1;
    const int epochs = 500;
    NetworkTrainer::train(network, X_train, Y_train, epochs, learning_rate);

    // evaluate on the full set
    float accuracy = network.eval(X_train, Y_train);
    std::cout << "Training accuracy: " << accuracy << std::endl;

    return 0;
}