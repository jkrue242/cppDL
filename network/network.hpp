#ifndef NETWORK_HPP
#define NETWORK_HPP
#include <Eigen/Dense>
#include <vector>
#include "layer.hpp"
#include "functions.hpp"

//==============================================
class Network {
    public:
    //==============================================
    Network(std::vector<Layer> layers)
    : _layers(layers)
    , _outputs({})
    , _result(Eigen::VectorXd::Zero(1)) // result is just a 1 dimensional vector of 0s to start
    {}

    //==============================================
    // forward pass
    // performs forward pass for each layer sequentially where 
    // each layer output becomes the next layers input
    // performs softmax at the end for final result vector
    //==============================================
    Eigen::VectorXd forward(const Eigen::Ref<const Eigen::VectorXd>& x) {
        _outputs.clear(); // clear current outputs for clean passthru
        Eigen::VectorXd current_input = x;
        Eigen::VectorXd layer_output;
        
        // go through each layer in the list, and pass the output to the next one
        for (auto& layer : _layers) {
            layer_output = layer.forward(current_input); // forward pass at the current layer
            _outputs.push_back(layer_output);
            current_input = layer_output; // make the output the next input
        }
        
        // softmax to create a probability distribution
        _result = LayerwiseFunction::apply(layer_output, Softmax);
        return _result;
    }

    //==============================================
    // backward pass 
    // backpropagation by computing the gradients at each layer
    //==============================================
    void backward() {
        
    }

    private:
    std::vector<Layer> _layers;
    std::vector<Eigen::VectorXd> _outputs;
    Eigen::VectorXd _result; 
};


#endif //NETWORK_HPP