//==============================================
// joseph krueger, 2025
//==============================================
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
    : _W(Eigen::MatrixXf::Random(M, N)) // randomly initialize weight matrix
    , _b(Eigen::VectorXf::Zero(M)) // bias set to 0 vector
    , _x(Eigen::VectorXf::Zero(N)) // input set to 0
    , _z(Eigen::VectorXf::Zero(M)) // output before activation. store for gradient computations
    , _y(Eigen::VectorXf::Zero(M)) // output after activation
    , _dLdW(Eigen::MatrixXf::Zero(M, N)) // gradient w.r.t. weight matrix
    , _dLdb(Eigen::VectorXf::Zero(M)) // gradient w.r.t. bias vector
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
        else if (activation_function == "Identity") {
            _activation_function = LayerwiseFunction::identity;
            _activation_derivative_function = LayerwiseFunction::identityDerivative;
            _derivative_input = Y;
        }
        else if (activation_function == "Softmax") {
            throw std::runtime_error("Softmax activation must be implemented via the Loss function (e.g., SoftmaxCrossEntropyLoss).");
        } 
        else {
            throw std::runtime_error("Invalid activation function. Expected one of 'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'Identity'");
        }
    }

    //==============================================
    // getters
    //==============================================
    Eigen::VectorXf get_x() { return _x; }
    Eigen::VectorXf get_z() { return _z; }
    Eigen::VectorXf get_y() { return _y; }
    Eigen::MatrixXf get_W() { return _W; }
    Eigen::VectorXf get_b() { return _b; }
    Eigen::VectorXf get_dldb() { return _dLdb; }
    Eigen::MatrixXf get_dldW() { return _dLdW; }


    //==============================================
    // update step
    // updates the weights and biases of the layer
    //==============================================
    void update(float lr) {
        _W.noalias() -= lr * _dLdW;
        _b.noalias() -= lr * _dLdb;
    }

    //==============================================
    // forward pass
    // takes in an input vector x that must match the number of columns of the weight matrix
    // returns an output vector after applying the activation function layer-wise
    // y = a(Wx+b)
    //==============================================
    Eigen::VectorXf forward(const Eigen::Ref<const Eigen::VectorXf>& x) {
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
    Eigen::VectorXf backward(const Eigen::Ref<const Eigen::VectorXf>& upstream_gradient) {
        assert(upstream_gradient.size() == _y.size() && "Upstream gradient size must match output size");

        const Eigen::VectorXf* derivative_input;
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
        
        Eigen::VectorXf dadx = _activation_derivative_function(*derivative_input);
        Eigen::VectorXf dLdz = upstream_gradient.cwiseProduct(dadx);
        Eigen::VectorXf dLdx = _W.transpose() * dLdz;
        Eigen::MatrixXf dLdW = dLdz * _x.transpose();
        Eigen::VectorXf dLdb = dLdz;
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

    using ActivationFunction = std::function<Eigen::VectorXf(const Eigen::Ref<const Eigen::VectorXf>&)>;
    using ActivationDerivativeFunction = std::function<Eigen::VectorXf(const Eigen::Ref<const Eigen::VectorXf>&)>;
    enum DerivativeInput { Z, Y };

    ActivationFunction _activation_function;
    ActivationDerivativeFunction _activation_derivative_function;
    DerivativeInput _derivative_input;

    Eigen::MatrixXf _W; // weight matrix
    Eigen::VectorXf _b; // bias vector
    Eigen::VectorXf _x; // input vector
    Eigen::VectorXf _z; // linear output: Wx+b
    Eigen::VectorXf _y; // activated output: a(Wx+b)
    Eigen::MatrixXf _dLdW; // gradient w.r.t. weights: dL/dW
    Eigen::VectorXf _dLdb; // gradient w.r.t. bias: dL/db 
    std::string _activation_name;
};


#endif // LAYER_HPP