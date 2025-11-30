#ifndef NETWORK_HPP
#define NETWORK_HPP
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "layer.hpp"
#include "functions.hpp"
#include "loss_layer.hpp"

//==============================================
class Network {
    public:
    //==============================================
    Network(std::vector<std::unique_ptr<Layer>> layers, std::unique_ptr<LossLayer> loss_function)
    : _layers(std::move(layers))
    , _loss_function(std::move(loss_function))
    , _y(Eigen::VectorXd::Zero(1)) 
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
    Eigen::VectorXd forward(const Eigen::Ref<const Eigen::VectorXd>& x) {
        _outputs.clear(); // clear current outputs for clean passthru
        Eigen::VectorXd curr_input = x;

        // iterate  through all the layers
        for (const auto& layer : _layers) {
            Eigen::VectorXd layer_output = layer->forward(curr_input);
            _outputs.push_back(layer_output);
            curr_input = layer_output;
        }

        _y = curr_input; // output of entire network
        return _y;
    }

    //==============================================
    // backpropagation
    // computes gradient of the loss with respect to 
    // the output and propagates it backwards
    //==============================================
    void backprop(const Eigen::Ref<const Eigen::VectorXd>& y_true) {
        if (_outputs.empty()) { // these get stored during forward pass 
            throw std::runtime_error("Must run forward() before backprop().");
        }        

        // softmax  TODO: i need to maybe make this more general
        Eigen::VectorXd y_softmax = LayerwiseFunction::softmax(_y);
        Eigen::VectorXd upstream_gradient = _loss_function->backward(y_softmax, y_true); // initial graident to be propagated
        
        for (int i = _layers.size()-1; i >= 0; i--) {
            _layers[i]->clear_grads(); // clear the gradients for the layer
            upstream_gradient = _layers[i]->backward(upstream_gradient); // perform backward pass 
        }
    }

    //==============================================
    // computes loss
    //==============================================
    double compute_loss(const Eigen::Ref<const Eigen::VectorXd>& y_true) {
        Eigen::VectorXd y_softmax = LayerwiseFunction::softmax(_y);
        return _loss_function->forward(y_softmax, y_true);
    }


    //==============================================
    // applies gradient updates
    //==============================================
    void apply_updates(double lr) {
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

    private:
    std::vector<std::unique_ptr<Layer>> _layers;
    std::unique_ptr<LossLayer> _loss_function;
    std::vector<Eigen::VectorXd> _outputs;
    Eigen::VectorXd _y; 
};


#endif //NETWORK_HPP