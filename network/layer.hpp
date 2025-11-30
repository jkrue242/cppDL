#ifndef LAYER_HPP
#define LAYER_HPP
#include <cmath>
#include <algorithm>
#include <cassert>
#include <Eigen/Dense>
#include <iostream>
#include "functions.hpp"

//==============================================
// Layer class 
// This models a linear layer. it is defined by a weight vector, w, 
// of arbitrary length and a bias scalar, b. It takes in a input vector of the same
// length as w, and performs a weighted sum with activation for the output:
// f(x) = a(W @ x + b)
// the size parameter of the layer defines the number of neurons it has. each neuron 
// represents an index into the weight vector. it takes in a vector input of size n and 
// returns an output vector of size n after forward pass.
//==============================================
class Layer {
    public:
    //==============================================
    Layer(int input_size, int neurons, Function activation = ReLU)
    : _W(Eigen::MatrixXd::Random(neurons, input_size)) // randomly initialize weight matrix
    , _b(Eigen::VectorXd::Zero(neurons)) // bias set to 0 vector
    , _input(Eigen::VectorXd::Zero(input_size)) // input set to 0
    , _output(Eigen::VectorXd::Zero(neurons)) // output set to 0 vector
    , _activation(activation)
    {}

    //==============================================
    // update step
    // updates the weights and biases of the layer
    //==============================================
    void update(const Eigen::Ref<const Eigen::MatrixXd>& W, Eigen::VectorXd b) {
        assert(b.size() == _W.rows() && "Bias vector size should match number of neurons (weight matrix rows).");
        _W = W;
        _b = b;
    }

    //==============================================
    // forward pass
    // takes in an input vector x that must match the number of columns of the weight matrix
    // returns an output vector after applying the activation function layer-wise
    //==============================================
    Eigen::VectorXd forward(const Eigen::Ref<const Eigen::VectorXd>& x) {
        assert(x.size() == _W.cols() && "Input vector size must match weight matrix columns");
        _input = x;
        Eigen::VectorXd linear_output = _W * _input + _b;
        _output = LayerwiseFunction::apply(linear_output, _activation);
        return _output;
    }

    private:
    Eigen::MatrixXd _W;
    Eigen::VectorXd _b;
    Eigen::VectorXd _input;
    Eigen::VectorXd _output;
    Function _activation;
};


#endif // LAYER_HPP