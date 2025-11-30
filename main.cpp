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


    // train 
    for (int i = 0; i < epochs; i++) {
        double epoch_loss = 0.0;
        for (int j = 0; j < num_samples; j++) {
            Eigen::VectorXd x_sample = X_train.col(j);
            Eigen::VectorXd y_sample = Y_train.col(j);

            network.clear_grads(); // clear gradients
            network.forward(x_sample); // forward pass 
            epoch_loss += network.compute_loss(y_sample); // compute loss 
            network.backprop(y_sample); // backprop
            network.apply_updates(learning_rate); // update gradients
        }

        double avg_loss = epoch_loss / num_samples;
        std::cout << "Epoch " << i+1 << "/" << epochs << " - Avg Loss: " << avg_loss << std::endl;
    }

    std::cout << "Training complete." << std::endl;
    std::cout << "\n--- Final Test on Sample 1 (Target: " << Y_train.col(0).transpose() << ") ---\n";
    Eigen::VectorXd final_output = network.forward(X_train.col(0));
    Eigen::VectorXd final_prediction = LayerwiseFunction::softmax(final_output);
    std::cout << "Raw Output:   " << final_output.transpose() << std::endl;
    std::cout << "Probabilities:" << final_prediction.transpose() << std::endl;

    return 0;
}