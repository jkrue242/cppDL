//==============================================
// joseph krueger, 2025
//==============================================
#ifndef SOFTMAX_CROSS_ENTROPY_LOSS_HPP
#define SOFTMAX_CROSS_ENTROPY_LOSS_HPP 
#include <Eigen/Dense>
#include "loss_layer.hpp" 
#include <limits>
#include <cmath>

//==============================================
// Cross entropy loss
// IMPORTANT: I named this SoftmaxCrossEntropyLoss because the
// backward pass assumes that the input has already had softmax applied to it
//==============================================
class SoftmaxCrossEntropyLoss: public LossLayer {
public: 

    //==============================================
    // sum( t_i * log(y_i) )
    // we assume that y_true are one-hot encoded labels
    //==============================================
    double forward(const Eigen::Ref<const Eigen::VectorXd>& y_pred, const Eigen::Ref<const Eigen::VectorXd>& y_true) {
        double loss = 0.0;
        double e = 1e-12; // for numerical stability
        for (int i = 0; i < y_pred.size(); i++) {
            loss += (y_true(i) * std::log(std::max(y_pred(i), e)));
        }
        return -loss;
    }   

    //==============================================
    // gradient assuming softmax
    //==============================================
    Eigen::VectorXd backward(const Eigen::Ref<const Eigen::VectorXd>& y_pred, const Eigen::Ref<const Eigen::VectorXd>& y_true) {
        return y_pred - y_true;
    }
};


#endif // SOFTMAX_CROSS_ENTROPY_LOSS_HPP 