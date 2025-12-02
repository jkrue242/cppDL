//==============================================
// joseph krueger, 2025
//==============================================
#ifndef LAYER_HPP
#define LAYER_HPP
#include <Eigen/Dense>

//==============================================
// abstract Layer class 
//==============================================
class Layer {
public:  
    virtual ~Layer() = default;
    virtual Eigen::VectorXf forward(const Eigen::Ref<const Eigen::VectorXf>& x) = 0;
    virtual Eigen::VectorXf backward(const Eigen::Ref<const Eigen::VectorXf>& upstream_gradient) = 0;
    virtual void update(float lr) = 0;
    virtual void clear_grads() = 0;
    virtual Eigen::VectorXf get_z() = 0;
};

#endif // LAYER_HPP