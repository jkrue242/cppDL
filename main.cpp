#include "network/network.hpp"
#include "network/functions.hpp"
#include "network/layer.hpp"
#include "network/loss_layer.hpp"
#include "network/softmax_cross_entropy_loss.hpp"
#include "Eigen/Dense"
#include <iostream>
#include <random>
#include <memory>
#include <vector>

// random number generator 
std::random_device rd;
std::mt19937 rand_engine(rd());
std::uniform_real_distribution<double> distr(-1, 1);

// training hyperparameters
const double learning_rate = 0.01;
const int epochs = 100;
const int features = 2;
const int classes = 2;

int main() {

    // dataset
    Eigen::MatrixXd X_train(features, 4);
    X_train << 0.0, 1.0, 0.0, 1.0, 
               0.0, 0.0, 1.0, 1.0; 

    Eigen::MatrixXd Y_train(classes, 4);
    Y_train << 1.0, 0.0, 0.0, 1.0, 
                0.0, 1.0, 1.0, 0.0;

    int num_samples = X_train.cols();

    // build the network
    std::vector<std::unique_ptr<Layer>> layers;
    layers.push_back(std::make_unique<Layer>(features, 16, "ReLU"));
    layers.push_back(std::make_unique<Layer>(16, 16, "Tanh"));
    layers.push_back(std::make_unique<Layer>(16, classes, "Sigmoid"));
    std::unique_ptr<LossLayer> loss_layer = std::make_unique<SoftmaxCrossEntropyLoss>();
    Network network(std::move(layers), std::move(loss_layer));

    std::cout << "Starting training for " << epochs << " epochs..." << std::endl;
    std::cout << "Data size: " << num_samples << " samples." << std::endl;
}