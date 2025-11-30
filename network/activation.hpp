// activation.hpp
#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP
#include <algorithm>
#include <math.h>

//==============================================
// Activation class
// collection of activation functions
//==============================================
class Activation {
public:
    //==============================================
    // RELU activation
    // f(x) = max(0, x)
    //==============================================
    static double relu(const double& x) {
        return std::max(0.0, x);
    }

    //==============================================
    // Sigmoid activation
    // f(x) = 1/(1 + e^-x)
    //==============================================
    static double sigmoid(const double& x) {
        return 1.0/(1.0 + std::exp(-x));
    }
};

#endif // ACTIVATION_HPP