#pragma once
#include <Eigen/Dense>
#include "tensor.hpp"
#include <memory>
#include <vector>

class Embedding {
public:
    Embedding(int vocab_size, int embed_dim, int max_seq, bool sinusoidal = false);
    
    std::shared_ptr<Tensor> forward(const std::vector<int>& input_indices);
    
    std::vector<std::shared_ptr<Tensor>> parameters();
    void zero_grad();
    void get() const;

private:
    int vocab_size;
    int embed_dim;
    int max_seq;
    bool use_sinusoidal;
    
    // Token embeddings - always trainable
    std::shared_ptr<Tensor> token_embeddings;
    
    // Positional embeddings - depends on sinusoidal flag
    std::shared_ptr<Tensor> positional_embeddings_tensor;  // Trainable (when sinusoidal=false)
    Eigen::MatrixXd positional_embeddings_matrix;          // Fixed (when sinusoidal=true)
    
    Eigen::MatrixXd positional_embedding(int max_seq, int embed_dim);
};