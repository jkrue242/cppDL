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
    {}
private:  
    Eigen::MatrixXf _W;
};


#endif // EMBEDDING_HPP