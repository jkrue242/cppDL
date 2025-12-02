//==============================================
// joseph krueger, 2025
//==============================================
#ifndef LAYER_HPP
#define LAYER_HPP
#include <Eigen/Dense>
#include <stdexcept>

//==============================================
// abstract Layer class 
//==============================================
class Layer {
public:  
    virtual ~Layer() = default;

    // vector interfaces default
    virtual Eigen::VectorXf forward(const Eigen::Ref<const Eigen::VectorXf>& x) {
        return Eigen::VectorXf();
    }
    virtual Eigen::VectorXf backward(const Eigen::Ref<const Eigen::VectorXf>& upstream_gradient) {
        return Eigen::VectorXf();
    }
    virtual Eigen::VectorXf get_z() {
        return Eigen::VectorXf();
    }
    
    // matrix interfaces 
    virtual Eigen::MatrixXf forward(const Eigen::Ref<const Eigen::MatrixXf>& x) {
        return Eigen::MatrixXf();
    }
    virtual Eigen::MatrixXf backward(const Eigen::Ref<const Eigen::MatrixXf>& upstream_gradient) {
        return Eigen::MatrixXf();
    }
    virtual Eigen::MatrixXf get_z_matrix() {
        throw std::runtime_error("get_z_matrix not implemented for this layer type");
    }
    
    // common methods
    virtual void update(float lr){}
    virtual void clear_grads(){}
};

#endif // LAYER_HPP