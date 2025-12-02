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
    
    // vector interfaces
    virtual float forward(const Eigen::Ref<const Eigen::VectorXf>& y_pred, const Eigen::Ref<const Eigen::VectorXf>& y_true) { return 0.0; }
    virtual Eigen::VectorXf backward(const Eigen::Ref<const Eigen::VectorXf>& y_pred, const Eigen::Ref<const Eigen::VectorXf>& y_true) { return Eigen::VectorXf::Zero(y_pred.size()); }
    
    // matrix interfaces
    virtual float forward(const Eigen::Ref<const Eigen::MatrixXf>& y_pred, const Eigen::Ref<const Eigen::MatrixXf>& y_true) { return 0.0; }
    virtual Eigen::MatrixXf backward(const Eigen::Ref<const Eigen::MatrixXf>& y_pred, const Eigen::Ref<const Eigen::MatrixXf>& y_true) { return Eigen::MatrixXf::Zero(y_pred.rows(), y_pred.cols()); }
};

#endif // LOSS_HPP