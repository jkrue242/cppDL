#include "network/network.hpp"
#include "Eigen/Dense"
#include <iostream>
#include <random>
#include <vector>

// random number generator 
std::random_device rd;
std::mt19937 rand_engine(rd());
std::uniform_real_distribution<double> distr(-1, 1);

int main() {
    int n = 20; // 1x20 input
    Eigen::VectorXd x(n);

    for (int i=0; i<n; i++) {
        x(i) = distr(rand_engine);
    }
    
    std::vector<Layer> layers = {
        Layer(n, 32, ReLU),     
        Layer(32, 64, Sigmoid),
        Layer(64, 32, Sigmoid),
        Layer(32, 4, Sigmoid)
    };
    
    Network network(layers);

    // forward pass through the whole network
    Eigen::VectorXd result = network.forward(x);
    std::cout << "Forward pass result: " << result.transpose() << std::endl;
}