#include "dataset_util.h"
#include <string>
#include <iostream>
#include <set>
#include <vector>
#include <cstring>
#include "tokenizer.hpp"
#include <Eigen/Dense>

float train_split = 0.9;

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

    // create tokenizer
    std::set<char> character_set(dataset_array, dataset_array + buffer_length);
    std::vector<char> character_vec(character_set.begin(), character_set.end());
    Tokenizer tokenizer(character_vec);
    
    // encode the dataset
    Eigen::VectorXi dataset_encoded = tokenizer.encode(dataset_array);
    for (int i = 0; i < dataset_encoded.size(); i++) {
        std::cout << dataset_encoded(i);
    }

    // train/val split

    return 0;
}