#include "neuron.hpp"
#include "Eigen/Dense"
#include <iostream>

int main() {
    Eigen::VectorXd weights(3);
    weights << 3.0, 5.0, 2.3;

    double bias = 1.3;

    Eigen::VectorXd x(3);
    x << 1.0, 1.0, 1.0;
    
    // create neuron
    Neuron n = Neuron(weights, bias);
    double forward = n.forward(x);
    std::cout << "Forward pass: " << forward << std::endl;
}