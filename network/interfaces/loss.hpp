//==============================================
// joseph krueger, 2025
//==============================================
#ifndef LOSS_HPP
#define LOSS_HPP 
#include <Eigen/Dense>

//==============================================
// interface for loss layers 
//==============================================
class Loss {
public: 
    virtual ~Loss() = default;
    virtual float forward(const Eigen::Ref<const Eigen::VectorXf>& y_pred, const Eigen::Ref<const Eigen::VectorXf>& y_true) = 0;
    virtual Eigen::VectorXf backward(const Eigen::Ref<const Eigen::VectorXf>& y_pred, const Eigen::Ref<const Eigen::VectorXf>& y_true) = 0;
};

#endif // LOSS_HPP