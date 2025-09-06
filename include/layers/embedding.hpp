#pragma once
#include "tensor.hpp"
#include <Eigen/Dense>
#include <iostream>

class Embedding {
public:
    // vocab_size = number of tokens
    // d_model = embedding dimension
    // max_seq_len = maximum sequence length
    Embedding(int vocab_size, int embed_dim, int max_seq,bool sinusoidal);

    // Forward pass: input_indices shape = (seq_len)
    // Returns: (seq_len, d_model)
    Tensor forward(const TensorInt& input_indices);

private:
    Tensor token_embeddings;     // vocab_size x d_model
    Tensor positional_embeddings; // max_seq_len x d_model
    int vocab_size;
    int embed_dim;
    int max_seq;

    Tensor positional_embedding(int max_seq, int embed_dim);

};
