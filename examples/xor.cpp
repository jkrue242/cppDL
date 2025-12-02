//==============================================
// joseph krueger, 2025
//==============================================
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

//==============================================
// This is an example of training a network to 
// learn the "XOR" function
//==============================================

// random number generator 
std::random_device rd;
std::mt19937 rand_engine(rd());
std::uniform_real_distribution<float> distr(-1, 1);

// training hyperparameters
const float learning_rate = 0.1;
const int epochs = 500;
const int features = 2;
const int classes = 2;

// gets the class with highest output
int get_predicted_class(const Eigen::VectorXf& probabilities) {
    int max_index;
    probabilities.maxCoeff(&max_index);
    return max_index;
}

//==============================================
// training loop
//==============================================
void train(Network& network, const Eigen::Ref<const Eigen::MatrixXf>& X, const Eigen::Ref<const Eigen::MatrixXf>& y, bool verbose = true) {
    int num_samples = X.cols();
    for (int i = 0; i < epochs; i++) {
        float epoch_loss = 0.0;

        for (int j = 0; j < num_samples; j++) {
            Eigen::VectorXf x_sample = X.col(j);
            Eigen::VectorXf y_sample = y.col(j);

            network.forward(x_sample);                     // forward pass 
            epoch_loss += network.compute_loss(y_sample);  // compute loss 
            network.backprop(y_sample);                    // backprop

            network.apply_updates(learning_rate);          // update parameters
            network.clear_grads();                         // clear intermediate gradients
        }

        // print loss if we are in verbose mode
        if (verbose) {
            float avg_loss = epoch_loss / num_samples;
            std::cout << "Epoch " << i+1 << "/" << epochs << " - Avg Loss: " << avg_loss << std::endl;
        }
    }
}

//==============================================
// driver
//==============================================
int main() {

    // XOR dataset
    Eigen::MatrixXf X_train(features, 4);
    X_train << 0.0, 1.0, 0.0, 1.0, 
               0.0, 0.0, 1.0, 1.0; 

    // one hot encoded
    Eigen::MatrixXf Y_train(classes, 4);
    Y_train << 1.0, 0.0, 0.0, 1.0, 
                0.0, 1.0, 1.0, 0.0;

    int num_samples = X_train.cols();

    // build the network
    std::vector<std::unique_ptr<Layer>> layers;
    layers.push_back(std::make_unique<Layer>(features, 16, "ReLU"));
    layers.push_back(std::make_unique<Layer>(16, 16, "Sigmoid"));
    layers.push_back(std::make_unique<Layer>(16, classes, "Identity"));
    std::unique_ptr<LossLayer> loss_layer = std::make_unique<SoftmaxCrossEntropyLoss>();
    Network network(std::move(layers), std::move(loss_layer));

    // training loop
    train(network, X_train, Y_train);

    std::cout << "\n============================================\n";
    std::cout << "Training results:";

    int correct_predictions = 0;
    
    // Test on all training samples
    for (int j = 0; j < num_samples; j++) {
        Eigen::VectorXf x_sample = X_train.col(j);
        Eigen::VectorXf y_true = Y_train.col(j);

        // result is the forward pass combined with softmax for probabiltiy
        Eigen::VectorXf final_output = network.forward(x_sample);
        Eigen::VectorXf final_prediction = LayerwiseFunction::softmax(final_output);

        // see if class prediction was correct
        int predicted_class = get_predicted_class(final_prediction);
        int true_class = get_predicted_class(y_true);
        bool is_correct = (predicted_class == true_class);

        if (is_correct) {
            correct_predictions++;
        }
    }

    float accuracy = static_cast<float>(correct_predictions) / num_samples;
    std::cout << "\n--------------------------------------------\n";
    std::cout << "Total Correct: " << correct_predictions << "/" << num_samples << std::endl;
    std::cout << "Final Accuracy: " << (accuracy * 100.0) << "%\n";
    std::cout << "--------------------------------------------\n";
    return 0;
}