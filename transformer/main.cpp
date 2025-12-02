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

    // create tokenizer and encode the dataset
    std::set<char> character_set(dataset_array, dataset_array + buffer_length);
    std::vector<char> character_vec(character_set.begin(), character_set.end());
    Tokenizer tokenizer(character_vec);
    Eigen::VectorXi dataset_encoded = tokenizer.encode(dataset_array);

    // train/val split
    int train_samples = dataset_encoded.size() * train_split;
    int val_samples = dataset_encoded.size() - train_samples;
    Eigen::VectorXi train_set = dataset_encoded.head(train_samples);
    Eigen::VectorXi val_set = dataset_encoded.tail(val_samples);

    std::cout << "Train size: " << train_set.size() << "   [" << train_set.size() << "/" << dataset_encoded.size() << " = " << 100* train_set.size()/dataset_encoded.size() << "%]" << std::endl;
    std::cout << "Val size: " << val_set.size()  << "   [" << val_set.size() << "/" << dataset_encoded.size() << " = " << 100* val_set.size()/dataset_encoded.size() << "%]" << std::endl; 
    return 0;
}