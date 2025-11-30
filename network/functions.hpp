#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
#include <algorithm>
#include <cmath>
#include <cassert>
#include <Eigen/Dense>

//==============================================
// list of functions that can be applied layer-wise i.e. applied to 
// each neuron in the layer
//==============================================
enum Function {
    ReLU, Sigmoid, Softmax // will continue to add to this
};

//==============================================
// static class  for applying any of the functions
// in the Function enum
//==============================================
class LayerwiseFunction {
public:

    //==============================================
    // applies any of the layer wise functions from the Function enum
    //==============================================
    static Eigen::VectorXd apply(const Eigen::Ref<const Eigen::VectorXd>& x, Function function) {
        switch(function) {
            case ReLU: return _relu(x);
            case Sigmoid: return _sigmoid(x);
            case Softmax: return _softmax(x);
            default: return _relu(x); // default to relu
        }
    }

private:

    //==============================================
    // relu
    //==============================================
    static Eigen::VectorXd _relu(const Eigen::Ref<const Eigen::VectorXd>& x) {
        return x.cwiseMax(0.0);
    }

    //==============================================
    // sigmoid
    //==============================================
    static Eigen::VectorXd _sigmoid(const Eigen::Ref<const Eigen::VectorXd>& x) {
        Eigen::VectorXd result(x.size());
        for (int i = 0; i < x.size(); ++i) {
            double val = x(i);

            // for stability reasons https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-python
            if (val > 0) {
                result(i) = 1.0 / (1.0 + std::exp(-val));
            } else {
                result(i) = std::exp(val) / (1.0 + std::exp(val));
            }
        }
        return result;
    }

    //==============================================
    // softmax
    //==============================================
    static Eigen::VectorXd _softmax(const Eigen::Ref<const Eigen::VectorXd>& x) {
        double sum = 0.0;
        for (int i = 0; i < x.size(); i++) {
            sum += std::exp(x(i));
        }

        Eigen::VectorXd output(x.size()); // not sure if i should be making a new object for this or not
        for (int j = 0; j < x.size(); j++) {
            output(j) = std::exp(x(j)) / sum;
        }

        return output;
    }
};

#endif // FUNCTION_HPP