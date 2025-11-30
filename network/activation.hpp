// activation.hpp
#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP
#include <algorithm>
#include <math.h>

//==============================================
enum Activation {
    ReLU, Sigmoid // will add to this later
};

//==============================================
class ActivationFunction {
public:
    static double apply(double x, Activation activation) {
        switch(activation) {
            case ReLU: return _relu(x);
            case Sigmoid: return _sigmoid(x);
            default: return _relu(x); // default to relu
        }
    }

private:
    static double _relu(const double x) {
        return std::max(0.0, x);
    }

    static double _sigmoid(const double x) {
        return 1.0/(1.0 + std::exp(-x));
    }
};

#endif // ACTIVATION_HPP