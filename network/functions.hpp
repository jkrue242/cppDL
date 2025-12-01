//==============================================
// joseph krueger, 2025
//==============================================
#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
#include <algorithm>
#include <cmath>
#include <cassert>
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>

//==============================================
// static class with support for a variety of 
// layer-wise functions. mostly for activation functions. also
// has their derivatives
//==============================================
class LayerwiseFunction {
public:

    //==============================================
    // relu
    // f(x) = max(0, x)
    //==============================================
    static Eigen::VectorXd relu(const Eigen::Ref<const Eigen::VectorXd>& x) {
        return x.cwiseMax(0.0);
    }

    //==============================================
    // sigmoid
    // f(x) = 1 / (1 + e^-x)
    //==============================================
    static Eigen::VectorXd sigmoid(const Eigen::Ref<const Eigen::VectorXd>& x) {
        return (1.0 + (-x.array()).exp()).inverse().matrix();
    }

    //==============================================   
    // tanh
    //==============================================
    static Eigen::VectorXd tanh(const Eigen::Ref<const Eigen::VectorXd>& x) {
        return x.array().tanh().matrix();
    }

    //==============================================
    // softmax
    //==============================================
    static Eigen::VectorXd softmax(const Eigen::Ref<const Eigen::VectorXd>& x) {
        double max = x.maxCoeff();
        Eigen::VectorXd exps = (x.array() - max).exp().matrix(); // for numerical stability
        return exps / exps.sum();
    }

    //==============================================
    // sigmoid derivative
    // f'(x) = f(x) * (1 - f(x))
    // here we call it y as it takes in the output of the activation
    //==============================================
    static Eigen::VectorXd sigmoidDerivative(const Eigen::Ref<const Eigen::VectorXd>& y) {
        return y.cwiseProduct(Eigen::VectorXd::Ones(y.size()) - y);
    }

    //==============================================
    // relu derivative
    // f'(x) = { 0: x <= 0
    //           1: x > 0 }
    //==============================================
    static Eigen::VectorXd reluDerivative(const Eigen::Ref<const Eigen::VectorXd>& x) {
        return (x.array() > 0.0).cast<double>().matrix();
    }

    //==============================================
    // tanh derivative
    // f'(x) = 1 - f(x)^2
    // same as with sigmoid, we call it y as it is the output after activation
    //==============================================
    static Eigen::VectorXd tanhDerivative(const Eigen::Ref<const Eigen::VectorXd>& y) {
        return (1.0 - y.array().square()).matrix();
    }

    //==============================================
    // general softmax derivative
    //==============================================
    static Eigen::VectorXd softmaxDerivative(const Eigen::Ref<const Eigen::VectorXd>& output, const Eigen::Ref<const Eigen::VectorXd>& loss_gradient) {
        double dot_product = loss_gradient.dot(output);
        return output.cwiseProduct(loss_gradient - Eigen::VectorXd::Constant(loss_gradient.size(), dot_product));
    }

    //==============================================
    // softmax derivative if we are using cross entropy loss
    //==============================================
    static Eigen::VectorXd softmaxCrossEntropyDerivative(const Eigen::Ref<const Eigen::VectorXd>& output, const Eigen::Ref<const Eigen::VectorXd>& target) {
        return output - target;
    }

    //==============================================
    // identity function
    //==============================================
    static Eigen::VectorXd identity(const Eigen::Ref<const Eigen::VectorXd>& x) {
        return x;
    }

    //==============================================
    // identity derivative
    //==============================================
    static Eigen::VectorXd identityDerivative(const Eigen::Ref<const Eigen::VectorXd>& y) {
        return Eigen::VectorXd::Constant(y.size(), 1.0);
    }
};

#endif // FUNCTION_HPP