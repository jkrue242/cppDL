#include "layer.hpp"
#include "activation.hpp"
#include "Eigen/Dense"
#include <iostream>
#include <random>

// random number generator 
std::random_device rd;
std::mt19937 rand_engine(rd());
std::uniform_real_distribution<double> distr(1, 10);

int main() {
    int n = 3;
    Eigen::VectorXd weights(n);
    Eigen::VectorXd x(n);

    for (int i=0; i<n; i++) {
        weights(i) = distr(rand_engine); 
        x(i) = distr(rand_engine);
    }

    double bias = distr(rand_engine);
    
    // create single layer network with n neurons
    Layer network = Layer(n, ReLU);
    network.update(weights, bias);
    double result = network.forward(x);
    std::cout << "Forward pass result: " << result << std::endl;
}