#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP 
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "network/interfaces/layer.hpp"

//==============================================
// embedding class 
// analogous to torch.nn.Embedding
// Embedding is a large matrix that is defined by 
// the vocab size (V) and the embedding dimension (D). Each 
// row in the weight matrix W represents a token id. The number of 
// token IDs is 
//==============================================
class Embedding: public Layer {
public:  
    Embedding(int V, int D) {

    }
private:  
    Eigen::MatrixXf _W;
};


#endif // EMBEDDING_HPP