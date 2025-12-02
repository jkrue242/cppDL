//==============================================
// joseph krueger, 2025
//==============================================
#ifndef NETWORK_HPP
#define NETWORK_HPP
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "interfaces/layer.hpp"
#include "functions.hpp"
#include "interfaces/loss.hpp"

//==============================================
// Network class 
// models a general neural network of an arbitrary depth. It is defined 
// by a vector of layers (Layer) and a loss function (Loss)
//==============================================
class Network {
    public:
    //==============================================
    Network(std::vector<std::unique_ptr<Layer>> layers, std::unique_ptr<Loss> loss_function)
    : _layers(std::move(layers))
    , _loss_function(std::move(loss_function))
    , _y(Eigen::VectorXf::Zero(1)) 
    {
        if (_layers.empty()) {
            throw std::runtime_error("Network must contain at least one layer.");
        }
    }

    //==============================================
    // forward pass
    // performs forward pass for each layer sequentially where 
    // each layer output becomes the next layers input
    //==============================================
    Eigen::VectorXf forward(const Eigen::Ref<const Eigen::VectorXf>& x) {
        _outputs.clear(); // clear current outputs for clean passthru
        Eigen::VectorXf curr_input = x;

        // iterate  through all the layers
        for (int i = 0; i < _layers.size(); i++) {
            auto& layer = _layers[i];

            // output for the layer 
            Eigen::Ref<const Eigen::VectorXf> input_ref = curr_input;
            Eigen::VectorXf layer_output = layer->forward(input_ref);
            _outputs.push_back(layer_output);
            
            // if we are on the last layer, we want the linear output only
            if (i == _layers.size()-1) {
                _y = layer->get_z();
            }
            else {
                curr_input = layer_output;
            }
        }
        return _y;
    }

    //==============================================
    // backpropagation
    // computes gradient of the loss with respect to 
    // the output and propagates it backwards. starts with softmax
    //==============================================
    void backprop(const Eigen::Ref<const Eigen::VectorXf>& y_true) {
        if (_outputs.empty()) { // these get stored during forward pass 
            throw std::runtime_error("Must run forward() before backprop().");
        }        

        // We always perform softmax at the end. I should probably make this more general at somepoint
        Eigen::VectorXf y_softmax = LayerwiseFunction::softmax(_y);
        Eigen::VectorXf upstream_gradient = _loss_function->backward(y_softmax, y_true); // initial graident to be propagated
        
        for (int i = _layers.size()-1; i >= 0; i--) {
            _layers[i]->clear_grads(); // clear the gradients for the layer
            Eigen::Ref<const Eigen::VectorXf> gradient_ref = upstream_gradient;
            upstream_gradient = _layers[i]->backward(gradient_ref); // perform backward pass 
        }
    }

    //==============================================
    // computes loss
    //==============================================
    float compute_loss(const Eigen::Ref<const Eigen::VectorXf>& y_true) {
        Eigen::VectorXf y_softmax = LayerwiseFunction::softmax(_y);
        return _loss_function->forward(y_softmax, y_true);
    }


    //==============================================
    // applies gradient updates
    //==============================================
    void apply_updates(float lr) {
        for (const auto& layer : _layers) {
            layer->update(lr);
        }
    }

    //==============================================
    // clears all the gradients across the layers
    //==============================================
    void clear_grads() {
        for (const auto& layer: _layers) {
            layer->clear_grads();
        }
    }

    //==============================================
    // returns the class from softmax output
    //==============================================
    static int get_predicted_class(const Eigen::VectorXf& probabilities) {
        int max_index;
        probabilities.maxCoeff(&max_index);
        return max_index;
    }

    //==============================================
    // evaluate the network on a given sample set.
    // returns the accuracy over the samples
    //==============================================
    float eval(const Eigen::Ref<const Eigen::MatrixXf>& X, const Eigen::Ref<const Eigen::MatrixXf>& y) {
        int samples = X.cols();
        int n_correct = 0;
        for (int i = 0; i < samples; i++) {
            Eigen::VectorXf sample = X.col(i);
            Eigen::VectorXf y_pred = forward(sample); // run sample through the network
            Eigen::VectorXf y_true = y.col(i);
            
            int true_class = get_predicted_class(y_true);
            int pred_class = get_predicted_class(y_pred);
            if (true_class == pred_class) {
                n_correct += 1;
            }
        }
        float acc = static_cast<float>(n_correct) / samples;
        return acc;
    } 

    private:
    std::vector<std::unique_ptr<Layer>> _layers;
    std::unique_ptr<Loss> _loss_function;
    std::vector<Eigen::VectorXf> _outputs;
    Eigen::VectorXf _y; 
};


#endif //NETWORK_HPP