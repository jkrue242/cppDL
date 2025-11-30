#ifndef LAYER_HPP
#define LAYER_HPP
#include <cmath>
#include <algorithm>
#include <cassert>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include "functions.hpp"

//==============================================
// Layer class 
// This models a linear layer. it is defined by a weight vector, w, 
// of arbitrary length and a bias scalar, b. It takes in an input vector x of 
// size N and an output vector y of size M where M is the number of neurons, such 
// that y = a(Wx + b)
//==============================================
class Layer {
    public:
    //==============================================
    Layer(int N, int M, std::string activation_function)
    : _W(Eigen::MatrixXd::Random(M, N)) // randomly initialize weight matrix
    , _b(Eigen::VectorXd::Zero(M)) // bias set to 0 vector
    , _x(Eigen::VectorXd::Zero(N)) // input set to 0
    , _z(Eigen::VectorXd::Zero(M)) // output before activation. store for gradient computations
    , _y(Eigen::VectorXd::Zero(M)) // output after activation
    , _dLdW(Eigen::MatrixXd::Zero(M, N)) // gradient w.r.t. weight matrix
    , _dLdb(Eigen::VectorXd::Zero(M)) // gradient w.r.t. bias vector
    , _activation_name(activation_function)
    {
        if (activation_function == "ReLU") {
            _activation_function = LayerwiseFunction::relu;
            _activation_derivative_function = LayerwiseFunction::reluDerivative;
            _derivative_input = Z;
        }
        else if (activation_function == "Sigmoid") {
            _activation_function = LayerwiseFunction::sigmoid;
            _activation_derivative_function = LayerwiseFunction::sigmoidDerivative;
            _derivative_input = Y;

        }
        else if (activation_function == "Tanh") {
            _activation_function = LayerwiseFunction::tanh;
            _activation_derivative_function = LayerwiseFunction::tanhDerivative;
            _derivative_input = Y;
        }
        else if (activation_function == "Softmax") {
            throw std::runtime_error("Softmax activation must be implemented via the Loss function (e.g., SoftmaxCrossEntropyLoss).");
        } 
        else {
            throw std::runtime_error("Invalid activation function. Expected one of 'ReLU', 'Sigmoid', 'Tanh', 'Softmax'");
        }
    }

    //==============================================
    // update step
    // updates the weights and biases of the layer
    //==============================================
    void update(double lr) {
        _W.noalias() -= lr * _dLdW;
        _b.noalias() -= lr * _dLdb;
    }

    //==============================================
    // forward pass
    // takes in an input vector x that must match the number of columns of the weight matrix
    // returns an output vector after applying the activation function layer-wise
    // y = a(Wx+b)
    //==============================================
    Eigen::VectorXd forward(const Eigen::Ref<const Eigen::VectorXd>& x) {
        assert(x.size() == _W.cols() && "Input vector size must match weight matrix columns");
        _x = x; // store input
        _z = _W * _x + _b; // store linear output
        _y = _activation_function(_z);
        return _y;
    }

    //==============================================
    // backward pass 
    // receives the gradient from the next layer w.r.t. the output of the network (dL/dy). This is
    // propagated backwards through the layer. gradients are computed w.r.t the layer input vector,
    // weight matrix, and bias vector via chain rule. final output is the gradient of the layer output
    // w.r.t the input vector (dL/dx) 
    //==============================================
    Eigen::VectorXd backward(const Eigen::Ref<const Eigen::VectorXd>& upstream_gradient) {
        assert(upstream_gradient.size() == _y.size() && "Upstream gradient size must match output size");

        const Eigen::VectorXd* derivative_input;
        switch(_derivative_input) {
            case(Z): 
                derivative_input = &_z;
                break;
            case(Y):
                derivative_input = &_y; 
                break;
            default:  
                throw std::logic_error("Unknown derivative input type. This should never happen");
        }
        
        Eigen::VectorXd dadx = _activation_derivative_function(*derivative_input);
        Eigen::VectorXd dLdz = upstream_gradient.cwiseProduct(dadx);
        Eigen::VectorXd dLdx = _W.transpose() * dLdz;
        Eigen::MatrixXd dLdW = dLdz * _x.transpose();
        Eigen::VectorXd dLdb = dLdz;
        _dLdW += dLdW;
        _dLdb += dLdb;
        return dLdx;
    }

    //==============================================
    // clear gradients 
    //==============================================
    void clear_grads() { 
        _dLdW.setZero();
        _dLdb.setZero();
    }

    private:

    using ActivationFunction = std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>&)>;
    using ActivationDerivativeFunction = std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>&)>;
    enum DerivativeInput { Z, Y };

    ActivationFunction _activation_function;
    ActivationDerivativeFunction _activation_derivative_function;
    DerivativeInput _derivative_input;

    Eigen::MatrixXd _W; // weight matrix
    Eigen::VectorXd _b; // bias vector
    Eigen::VectorXd _x; // input vector
    Eigen::VectorXd _z; // linear output: Wx+b
    Eigen::VectorXd _y; // activated output: a(Wx+b)
    Eigen::MatrixXd _dLdW; // gradient w.r.t. weights: dL/dW
    Eigen::VectorXd _dLdb; // gradient w.r.t. bias: dL/db 
    std::string _activation_name;
};


#endif // LAYER_HPP