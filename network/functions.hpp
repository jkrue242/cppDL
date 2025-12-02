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
    static Eigen::VectorXf relu(const Eigen::Ref<const Eigen::VectorXf>& x) {
        return x.cwiseMax(0.0);
    }

    //==============================================
    // sigmoid
    // f(x) = 1 / (1 + e^-x)
    //==============================================
    static Eigen::VectorXf sigmoid(const Eigen::Ref<const Eigen::VectorXf>& x) {
        return (1.0 + (-x.array()).exp()).inverse().matrix();
    }

    //==============================================   
    // tanh
    //==============================================
    static Eigen::VectorXf tanh(const Eigen::Ref<const Eigen::VectorXf>& x) {
        return x.array().tanh().matrix();
    }

    //==============================================
    // softmax
    //==============================================
    static Eigen::VectorXf softmax(const Eigen::Ref<const Eigen::VectorXf>& x) {
        float max = x.maxCoeff();
        Eigen::VectorXf exps = (x.array() - max).exp().matrix(); // for numerical stability
        return exps / exps.sum();
    }

    //==============================================
    // sigmoid derivative
    // f'(x) = f(x) * (1 - f(x))
    // here we call it y as it takes in the output of the activation
    //==============================================
    static Eigen::VectorXf sigmoidDerivative(const Eigen::Ref<const Eigen::VectorXf>& y) {
        return y.cwiseProduct(Eigen::VectorXf::Ones(y.size()) - y);
    }

    //==============================================
    // relu derivative
    // f'(x) = { 0: x <= 0
    //           1: x > 0 }
    //==============================================
    static Eigen::VectorXf reluDerivative(const Eigen::Ref<const Eigen::VectorXf>& x) {
        return (x.array() > 0.0).cast<float>().matrix();
    }

    //==============================================
    // tanh derivative
    // f'(x) = 1 - f(x)^2
    // same as with sigmoid, we call it y as it is the output after activation
    //==============================================
    static Eigen::VectorXf tanhDerivative(const Eigen::Ref<const Eigen::VectorXf>& y) {
        return (1.0 - y.array().square()).matrix();
    }

    //==============================================
    // general softmax derivative
    //==============================================
    static Eigen::VectorXf softmaxDerivative(const Eigen::Ref<const Eigen::VectorXf>& output, const Eigen::Ref<const Eigen::VectorXf>& loss_gradient) {
        float dot_product = loss_gradient.dot(output);
        return output.cwiseProduct(loss_gradient - Eigen::VectorXf::Constant(loss_gradient.size(), dot_product));
    }

    //==============================================
    // softmax derivative if we are using cross entropy loss
    //==============================================
    static Eigen::VectorXf softmaxCrossEntropyDerivative(const Eigen::Ref<const Eigen::VectorXf>& output, const Eigen::Ref<const Eigen::VectorXf>& target) {
        return output - target;
    }

    //==============================================
    // identity function
    //==============================================
    static Eigen::VectorXf identity(const Eigen::Ref<const Eigen::VectorXf>& x) {
        return x;
    }

    //==============================================
    // identity derivative
    //==============================================
    static Eigen::VectorXf identityDerivative(const Eigen::Ref<const Eigen::VectorXf>& y) {
        return Eigen::VectorXf::Constant(y.size(), 1.0);
    }
};

#endif // FUNCTION_HPP