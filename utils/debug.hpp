//==============================================
// joseph krueger, 2025
//==============================================
#ifndef DEBUG_HPP
#define DEBUG_HPP
#include <Eigen/Dense>
#include <iostream>

//==============================================
// print matrix
//==============================================
void print_matrix(const Eigen::Ref<const Eigen::MatrixXf>& matrix, std::string name = "", bool show_data = false) {
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Matrix: " << std::endl;
    if (name != "") {
        std::cout << "Name: " << name << std::endl;
    }
    std::cout << "Shape: " << matrix.rows() << "x" << matrix.cols() << std::endl;
    if (show_data) {
        std::cout << "Data: " << std::endl;
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                std::cout << matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "--------------------------------" << std::endl;
}

//==============================================
// print vector
//==============================================
void print_vector(const Eigen::Ref<const Eigen::VectorXf>& vector, std::string name = "", bool show_data = false) {
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Vector: " << std::endl;
    if (name != "") {
        std::cout << "Name: " << name << std::endl;
    }
    std::cout << "Shape: " << vector.size() << std::endl;
    if (show_data) {
        std::cout << "Data: " << std::endl;
        for (int i = 0; i < vector.size(); i++) {
            std::cout << vector(i) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------------------------" << std::endl;
}

#endif // DEBUG_HPP