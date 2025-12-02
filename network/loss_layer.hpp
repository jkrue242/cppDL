//==============================================
// joseph krueger, 2025
//==============================================
#ifndef LOSS_LAYER_HPP
#define LOSS_LAYER_HPP 
#include <Eigen/Dense>

//==============================================
// interface for loss layers 
//==============================================
class LossLayer {
public: 
    virtual ~LossLayer() = default;

    //==============================================
    // calculates loss 
    //==============================================
    virtual float forward(const Eigen::Ref<const Eigen::VectorXf>& y_pred, const Eigen::Ref<const Eigen::VectorXf>& y_true) = 0;

    //==============================================
    // calculates initial gradient for backpropagation
    //==============================================
    virtual Eigen::VectorXf backward(const Eigen::Ref<const Eigen::VectorXf>& y_pred, const Eigen::Ref<const Eigen::VectorXf>& y_true) = 0;

};

#endif // LOSS_LAYER_HPP