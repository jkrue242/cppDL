#ifndef NEURON_HPP
#define NEURON_HPP
#include <cmath>
#include <algorithm>
#include <cassert>
#include <Eigen/Dense>
#include <iostream>

//==============================================
// Neuron class 
//==============================================
class Neuron {
    public:
    //==============================================
    // Constructor
    //==============================================
    Neuron(const Eigen::Ref<const Eigen::VectorXd>& w, double b)
    : _w(w)
    , _b(b)
    , _input(Eigen::VectorXd::Zero(w.size()))
    , _output(0.0)
    {}

    //==============================================
    // forward pass
    //==============================================
    double forward(const Eigen::Ref<const Eigen::VectorXd>& x) {
        // x = w @ x + b 
        assert(x.size() == _w.size() && "Input vector size must match weight vector size");
        _input = x;
        _output = _w.dot(x) + _b;
        return _output;
    }

    private:
    Eigen::VectorXd _w;
    double _b;
    Eigen::VectorXd _input;
    double _output;
};


#endif // NEURON_HPP