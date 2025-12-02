//==============================================
// joseph krueger, 2025
//==============================================
#ifndef SOFTMAX_CROSS_ENTROPY_LOSS_HPP
#define SOFTMAX_CROSS_ENTROPY_LOSS_HPP 
#include <Eigen/Dense>
#include "interfaces/loss.hpp" 
#include <limits>
#include <cmath>

//==============================================
// Cross entropy loss
// IMPORTANT: I named this SoftmaxCrossEntropyLoss because the
// backward pass assumes that the input has already had softmax applied to it
//==============================================
class SoftmaxCrossEntropyLoss: public Loss {
public: 

    //==============================================
    // sum( t_i * log(y_i) )
    // we assume that y_true are one-hot encoded labels
    //==============================================
    float forward(const Eigen::Ref<const Eigen::VectorXf>& y_pred, const Eigen::Ref<const Eigen::VectorXf>& y_true) {
        float loss = 0.0;
        float e = 1e-12; // for numerical stability
        for (int i = 0; i < y_pred.size(); i++) {
            loss += (y_true(i) * std::log(std::max(y_pred(i), e)));
        }
        return -loss;
    }   

    //==============================================
    // gradient assuming softmax
    //==============================================
    Eigen::VectorXf backward(const Eigen::Ref<const Eigen::VectorXf>& y_pred, const Eigen::Ref<const Eigen::VectorXf>& y_true) {
        return y_pred - y_true;
    }
};


#endif // SOFTMAX_CROSS_ENTROPY_LOSS_HPP 