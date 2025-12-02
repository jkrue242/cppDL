//==============================================
// joseph krueger, 2025
//==============================================
#ifndef LOGITS_CROSS_ENTROPY_LOSS_HPP
#define LOGITS_CROSS_ENTROPY_LOSS_HPP
#include <Eigen/Dense>
#include "interfaces/loss.hpp"
#include <limits>
#include <cmath>

//==============================================
// I evenetually need to combine both this and the softmax_cross_entropy into one CrossEntropyLoss class.
//==============================================
class LogitsCrossEntropyLoss: public Loss {
public:

    //==============================================
    // Forward pass
    // average of -sum( Y * log(P) ) across the batch
    //==============================================
    float forward(const Eigen::Ref<const Eigen::MatrixXf>& y_pred, const Eigen::Ref<const Eigen::MatrixXf>& y_true) {
        int num_samples = y_pred.rows();

        // this is to prevent numerical overflow
        Eigen::VectorXf max_logits = y_pred.rowwise().maxCoeff();
        Eigen::MatrixXf shifted_logits = y_pred.colwise() - max_logits;

        Eigen::VectorXf log_sum_exp = (shifted_logits.array().exp().rowwise().sum()).log();
        Eigen::MatrixXf log_probs = y_pred.colwise() - (max_logits + log_sum_exp).array().matrix(); 
        Eigen::MatrixXf product = y_true.cwiseProduct(log_probs);
        
        float total_loss = -product.sum(); 
        float average_loss = total_loss / static_cast<float>(num_samples); 
        return average_loss;
    }

    //==============================================
    // backward pass 
    // gradient of the loss w.r.t the output logits
    //==============================================
    Eigen::MatrixXf backward(const Eigen::Ref<const Eigen::MatrixXf>& y_pred, const Eigen::Ref<const Eigen::MatrixXf>& y_true) {
        int num_samples = y_pred.rows();
        
        // same as in forward this is for stability
        Eigen::VectorXf max_logits = y_pred.rowwise().maxCoeff();
        Eigen::MatrixXf shifted_logits = y_pred.colwise() - max_logits;

        Eigen::MatrixXf P = shifted_logits.array().exp(); // probabilities of the logits
        Eigen::MatrixXf P_softmax = P.array().colwise() / P.rowwise().sum().array(); // softmax probs

        Eigen::MatrixXf grad = P_softmax - y_true; // matches softmax ce loss
        Eigen::MatrixXf grad_avg = grad / static_cast<float>(num_samples);
        return grad_avg;
    }
};

#endif // LOGITS_CROSS_ENTROPY_LOSS_HPP