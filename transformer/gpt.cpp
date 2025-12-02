//==============================================
// joseph krueger, 2025
//==============================================
#include "dataset_util.h"
#include <string>
#include <iostream>
#include <set>
#include <vector>
#include <random>
#include <cstring>
#include <tuple>
#include "tokenizer.hpp"
#include <Eigen/Dense>
#include "network/layers/embedding.hpp"
#include "utils/debug.hpp"

thread_local std::mt19937 generator{std::random_device{}()}; // random number generator


//==============================================
// retrieve a batch from the dataset
// a batch is a tuple of train and val data. the input is a 1d vector (the list of tokens)
// and the output is a tuple of matrices, which are MxN where M is the block size and N is the batch size
//==============================================
std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> get_batch(const Eigen::Ref<const Eigen::VectorXf>& data, int batch_size, int block_size) {
    if (data.size() < block_size + 1) {
        throw std::runtime_error("Dataset size is too small for the requested block size.");
    }
    
    int max_offset = data.size() - (block_size + 1);
    std::uniform_int_distribution<int> distrib(0, max_offset);
    
    
    // these will hold the batches 
    Eigen::MatrixXf X(block_size, batch_size); // MxN
    Eigen::MatrixXf y(block_size, batch_size); // MxN
    
    for (int i = 0; i < batch_size; i++) {
        int rand_start_index = distrib(generator);  // random offset into the data to grab segment block
        X.col(i) = data.segment(rand_start_index, block_size);
        y.col(i) = data.segment(rand_start_index + 1, block_size);
    }
    return std::make_tuple(X, y);
}


//==============================================
// driver 
//==============================================
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
    Eigen::VectorXf dataset_encoded = tokenizer.encode(dataset_array).cast<float>();

    // train/val split
    float train_split = 0.9;
    int train_samples = static_cast<int>(dataset_encoded.size() * train_split);
    int val_samples = dataset_encoded.size() - train_samples;
    Eigen::VectorXf train_set = dataset_encoded.head(train_samples);
    Eigen::VectorXf val_set = dataset_encoded.tail(val_samples);

    std::cout << "Train size: " << train_set.size() << "   [" << train_set.size() << "/" << dataset_encoded.size() << " = " << 100* train_set.size()/dataset_encoded.size() << "%]" << std::endl;
    std::cout << "Val size: " << val_set.size()  << "   [" << val_set.size() << "/" << dataset_encoded.size() << " = " << 100* val_set.size()/dataset_encoded.size() << "%]" << std::endl; 
    
    // get batches
    int batch_size = 4;
    int block_size = 8;

    std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> batch_tuple = get_batch(train_set, batch_size, block_size);
    Eigen::MatrixXf x_batch = std::get<0>(batch_tuple);
    Eigen::MatrixXf y_batch = std::get<1>(batch_tuple);

    // embedding 
    int vocab_size = tokenizer.get_vocab_size();
    int embedding_dim = 32; // embedding dimension
    Embedding embedding(vocab_size, embedding_dim);
    
    // run batch thru embedding 
    Eigen::MatrixXf x_embedding = embedding.forward(x_batch);

    print_matrix(x_batch, "x_batch", false);
    print_matrix(x_embedding, "x_embedding", false);

    

    return 0;
}