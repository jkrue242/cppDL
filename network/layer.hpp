#ifndef LAYER_HPP
#define LAYER_HPP
#include <cmath>
#include <algorithm>
#include <cassert>
#include <Eigen/Dense>
#include <iostream>
#include "activation.hpp"

//==============================================
// Layer class 
// This models a linear layer. it is defined by a weight vector, w, 
// of arbitrary length and a bias scalar, b. It takes in a input vector of the same
// length as w, and performs a weighted sum with activation for the output:
// f(x) = a(w @ x + b)
// the size parameter of the layer defines the number of neurons it has. each neuron 
// represents an index into the weight vector.
//==============================================
class Layer {
    public:
    //==============================================
    Layer(int size, Activation activation = ReLU)
    : _w(Eigen::VectorXd::Random(size)) // randomly initialize weights 
    , _b(0.0) // bias set to 0
    , _input(Eigen::VectorXd::Zero(size)) // input set to 0
    , _output(0.0) // output set to 0
    , _activation(activation)
    {}

    //==============================================
    // update step
    // updates the weights and biases of the layer
    //==============================================
    void update(const Eigen::Ref<const Eigen::VectorXd>& w, double b) {
        _w = w;
        _b = b;
    }

    //==============================================
    // forward pass
    // takes in an input vector x that must match the size of the weight vector
    //==============================================
    double forward(const Eigen::Ref<const Eigen::VectorXd>& x) {
        // x = w @ x + b 
        assert(x.size() == _w.size() && "Input vector size must match weight vector size");
        _input = x;
        _output = ActivationFunction::apply(_w.dot(x) + _b, _activation);
        return _output;
    }

    private:
    Eigen::VectorXd _w;
    double _b;
    Eigen::VectorXd _input;
    double _output;
    Activation _activation;
};


#endif // LAYER_HPP