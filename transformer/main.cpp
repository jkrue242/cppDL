#include "dataset_util.h"
#include <string>
#include <iostream>
#include <set>
#include <vector>
#include <cstring>
#include "tokenizer.hpp"

int main() {

    // download tiny shakespeare dataset to /tmp
    std::string path = "/tmp/tinyshakespeare.txt";
    std::string url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
    DatasetUtil::download(url, path);

    // load to array 
    std::string buffer = DatasetUtil::file_to_buffer(path);
    int buffer_length = buffer.length();
    char dataset_array[buffer_length + 1]; // null terminated so add 1
    strcpy(dataset_array, buffer.c_str());

    // unique character set
    std::set<char> character_set(dataset_array, dataset_array + buffer_length);
    std::vector<char> character_vec(character_set.begin(), character_set.end());
    std::cout << "Number of characters in vocabulary: " << character_vec.size() << std::endl;

    Tokenizer tokenizer(character_vec);
    std::string test = "Hi im joseph";
    std::vector<int> encoded = tokenizer.encode(test);
    std::string decoded = tokenizer.decode(encoded);

    std::cout << "Test string: " << test << std::endl;
    std::cout << "Encoded: ";
    for (int i = 0; i < encoded.size(); i++) {
        std::cout<<encoded.at(i);
    }
    std::cout <<std::endl;
    std::cout << "Decoded: " << decoded << std::endl;
    return 0;
}