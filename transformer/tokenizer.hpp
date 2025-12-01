#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP 
#include <string>
#include <set>
#include <vector> 
#include <map>

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
    // encode a string into list of ints
    //==============================================
    std::vector<int> encode(std::string str) {
        std::vector<int> encoded;
        for (int i = 0; i < str.size(); i++) {
            encoded.push_back(_char_to_int_map.at(str.at(i)));
        }
        return encoded;
    }

    //==============================================
    // decode a list of ints to a string
    //==============================================
    std::string decode(std::vector<int> ints) {
        std::string decoded;
        decoded.reserve(ints.size());
        for (int i : ints) {
            decoded += _characters[i];
        }
        return decoded;
    }

private:  
    std::vector<char> _characters;
    std::map<char, int> _char_to_int_map;
};

#endif // TOKENIZER_HPP