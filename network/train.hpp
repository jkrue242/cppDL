//==============================================
// joseph krueger, 2025
//==============================================
#ifndef TRAIN_HPP
#define TRAIN_HPP
#include "network.hpp"
#include <Eigen/Dense>


//==============================================
// basic trainer for a generic neural network
// ill probably need to add to this later
//==============================================
class NetworkTrainer {
public:
    static void train(Network& network, const Eigen::Ref<const Eigen::MatrixXf>& X, const Eigen::Ref<const Eigen::MatrixXf>& y, int epochs, float learning_rate) {
        int samples = X.cols();
        float total_loss = 0.0;

        // iterate over each epoch
        for (int i = 0; i < epochs; i++) {
            float epoch_loss = 0.0;

            // iterate over each sample
            for (int j = 0; j < samples; j++) {
                Eigen::VectorXf x_sample = X.col(j);
                Eigen::VectorXf y_sample = y.col(j);
    
                Eigen::VectorXf y_pred = network.forward(x_sample); // forward pass 
                epoch_loss += network.compute_loss(y_sample); // accumulate loss
                network.backprop(y_sample); // backpropagate the loss 
                network.apply_updates(learning_rate); // update the parameters of the network
                network.clear_grads(); // clear intermediate gradients
            }

            // print to console
            float avg_loss = epoch_loss / samples;
            std::cout << "Epoch " << i+1 << "/" << epochs << " - Avg Loss: " << avg_loss << std::endl;
        }
    }
};


#endif // TRAIN_HPP