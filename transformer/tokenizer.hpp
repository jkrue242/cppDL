//==============================================
// joseph krueger, 2025
//==============================================
#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP 
#include <string>
#include <set>
#include <vector> 
#include <map>
#include <Eigen/Dense>

//==============================================
// tokenizer class 
//==============================================
class Tokenizer {
public:

    //==============================================
    // constructor
    // initializes the token mappings
    //==============================================
    Tokenizer(std::vector<char> characters)
    : _characters(characters) 
    {
        for (int i = 0; i < characters.size(); i++) {
            _char_to_int_map[characters[i]] = i;
        }
    }

    //==============================================
    // encode a string into vector of ints
    //==============================================
    Eigen::VectorXi encode(std::string str) {
        Eigen::VectorXi encoded(str.size());
        for (int i = 0; i < str.size(); i++) {
            encoded(i) = _char_to_int_map.at(str.at(i));
        }
        return encoded;
    }

    //==============================================
    // decode a vector of ints to a string
    //==============================================
    std::string decode(Eigen::VectorXi ints) {
        std::string decoded;
        decoded.reserve(ints.size());
        for (int i = 0; i < ints.size(); i++) {
            int token = ints(i);
            decoded += _characters.at(token);
        }
        return decoded;
    }



private:  
    std::vector<char> _characters;
    std::map<char, int> _char_to_int_map;
};

#endif // TOKENIZER_HPP