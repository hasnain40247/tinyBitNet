#include "layers/embedding.hpp"
#include <random>
#include <cmath>

Embedding::Embedding(int vocab_size, int embed_dim, int max_seq, bool sinusoidal)
    : vocab_size(vocab_size), embed_dim(embed_dim), max_seq(max_seq) {  // Initialize member variables!

    float limit = std::sqrt(6.0f / (vocab_size + embed_dim));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-limit, limit);
    
    token_embeddings = Tensor(vocab_size, embed_dim);
    for (int i = 0; i < token_embeddings.size(); ++i) {
        token_embeddings.data()[i] = dist(gen);
    }
    
    if (sinusoidal) {
        positional_embeddings = positional_embedding(max_seq, embed_dim);
    } else {
        positional_embeddings = Tensor(max_seq, embed_dim);
        for (int i = 0; i < positional_embeddings.size(); i++) {
            positional_embeddings.data()[i] = dist(gen);
        }
    }
}

Tensor Embedding::forward(const TensorInt& input_indices) {
    int seq_len = input_indices.size();
    Tensor output(seq_len, embed_dim);  // Now embed_dim is properly initialized

    for (int i = 0; i < seq_len; ++i) {
        int token_idx = input_indices[i];
        output.row(i) = token_embeddings.row(token_idx) + positional_embeddings.row(i);
    }

    return output;
}

Tensor Embedding::positional_embedding(int max_seq, int embed_dim) {
    Tensor pe(max_seq, embed_dim);
    
    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < embed_dim; i++) {
            float angle = pos / std::pow(10000.0f, 2.0f * (i / 2) / embed_dim);

            if (i % 2 == 0)
                pe(pos, i) = std::sin(angle);
            else
                pe(pos, i) = std::cos(angle);
        }
    }
    
    return pe;
}