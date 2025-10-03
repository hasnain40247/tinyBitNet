#pragma once
#include <Eigen/Dense>
#include "../data/tensor.hpp"
#include <memory>
#include <vector>

class Embedding {
public:
    Embedding(int vocab_size, int embed_dim, int max_seq, bool sinusoidal = false);
    
    
    std::shared_ptr<Tensor> forward(const std::vector<int>& input_indices);
    
    std::vector<std::shared_ptr<Tensor>> parameters() const;
    void zero_grad();
   

    std::shared_ptr<Tensor> token_embeddings;
    std::shared_ptr<Tensor> positional_embeddings_tensor;  
    Eigen::MatrixXd positional_embeddings_matrix; 



    void get_parameters() const;
    void get_gradients() const;
    void get_layer_info() const;

private:
    int vocab_size;
    int embed_dim;
    int max_seq;
    bool use_sinusoidal;
    
    Eigen::MatrixXd positional_embedding(int max_seq, int embed_dim);

    void initialize_parameters();
};