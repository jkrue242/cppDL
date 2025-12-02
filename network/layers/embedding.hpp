//==============================================
// joseph krueger, 2025
//==============================================
#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP 
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "network/interfaces/layer.hpp"

//==============================================
// Embedding layer class 
// Embedding is a large matrix that is defined by 
// the vocab size (V) and the embedding dimension (D). Each 
// row in the weight matrix W represents a token id. 
//==============================================
class Embedding: public Layer {
public:  
    Embedding(int V, int D) 
    : _W(Eigen::MatrixXf::Random(V, D)) // randomly initialize weight matrix
    , _dLdW(Eigen::MatrixXf::Zero(V, D)) // gradient is same size as weight matrix
    {}

    //==============================================
    // forward pass 
    // x represents token indices of size TxN (block size x batch size)
    // the output is matrix of size (TxN, D) where D is the embeddding dimension
    // forward pass is basically a lookup operation
    //==============================================
    Eigen::MatrixXf forward(const Eigen::Ref<const Eigen::MatrixXf>& x) {
        _indices = x; // store for backward pass
        int T = x.rows(); //  block size
        int N = x.cols(); // batch size
        int D = _W.cols();  // embedding dimension

        Eigen::MatrixXf output(T*N, D); // output matrix
        for (int n = 0; n < N; n++) {
            for (int t = 0; t < T; t++) {
                int token_id = static_cast<int>(x.coeff(t, n));
                int output_index = n*T + t;
                output.row(output_index) = _W.row(token_id); // copy to output
            }
        }
        return output;
    }

    //==============================================
    // backward pass
    // takes in the upstream gradient (dL/dy) and 
    // accumulates the gradients into the weight matrix
    // returns the gradient
    //==============================================
    Eigen::MatrixXf backward(const Eigen::Ref<const Eigen::MatrixXf>& upstream_gradient) {
        int T = _indices.rows();
        int N = _indices.cols();
        int D = _W.cols();

        _dLdW.setZero(); // clear grad

        for (int n = 0; n < N; n++) {
            for (int t = 0; t < T; t++) {
                int token_id = static_cast<int>(_indices.coeff(t, n));
                int output_index = n*T + t;
                _dLdW.row(token_id) += upstream_gradient.row(output_index);
            }
        }
        return _dLdW;
    }

private:  
    Eigen::MatrixXf _W;
    Eigen::MatrixXf _dLdW;
    Eigen::MatrixXf _indices;
};


#endif // EMBEDDING_HPP