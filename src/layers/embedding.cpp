#include "layers/embedding.hpp"
#include <random>
#include <cmath>

/**
TODO: Do I worry about padding or should I wait till I'm implementing batches?
 */

/**
* @brief Constructor for the Embedding layer.
* 
* @param vocab_size  Number of unique tokens in the vocabulary.
* @param embed_dim   Dimensionality of the embeddings.
* @param max_seq     Maximum sequence length (used for positional embeddings).
* @param sinusoidal  If true, use sinusoidal positional embeddings; 
*                    otherwise, learnable positional embeddings.
*/
Embedding::Embedding(int vocab_size, int embed_dim, int max_seq, bool sinusoidal)
    : vocab_size(vocab_size), embed_dim(embed_dim), max_seq(max_seq) {  // Initialize member variables!

    float limit = std::sqrt(6.0f / (vocab_size + embed_dim));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-limit, limit);

    // Using Xavier's random initialization here.
    // Apparently this tries to make the variance scale of the weights as close to the inputs and the outputs. This avoids blowing or vanishing gradients.
    
    token_embeddings = Eigen::MatrixXd(vocab_size, embed_dim);
    for (int i = 0; i < token_embeddings.size(); ++i) {
        token_embeddings.data()[i] = dist(gen);
    }
    
    if (sinusoidal) {
        positional_embeddings = positional_embedding(max_seq, embed_dim);
    } else {
        positional_embeddings = Eigen::MatrixXd(max_seq, embed_dim);
        for (int i = 0; i < positional_embeddings.size(); i++) {
            positional_embeddings.data()[i] = dist(gen);
        }
    }
}


/**
* @brief Forward pass for the embedder.
* 
* Converts a sequence of token indices into embeddings and adds positional embeddings.
* 
* @param input_indices  Eigen::MatrixXd of token indices of shape [seq_len].
* @return Eigen::MatrixXd of embeddings of shape [seq_len, embed_dim].
*/
Eigen::MatrixXd Embedding::forward(const Eigen::VectorXi& input_indices) {
    int seq_len = input_indices.size();
    Eigen::MatrixXd output(seq_len, embed_dim);  // Now embed_dim is properly initialized

    for (int i = 0; i < seq_len; ++i) {
        int token_idx = input_indices[i];
        output.row(i) = token_embeddings.row(token_idx) + positional_embeddings.row(i);
    }

    return output;
}


/**
* @brief Generates positional embeddings.
* 
* Can be sinusoidal or learnable depending on the constructor.
* 
* @param max_seq   Maximum sequence length
* @param embed_dim Embedding dimension
* @return Eigen::MatrixXd of shape [max_seq, embed_dim]
*/
Eigen::MatrixXd Embedding::positional_embedding(int max_seq, int embed_dim) {
    Eigen::MatrixXd pe(max_seq, embed_dim);
    
    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < embed_dim; i++) {
            float angle = pos / std::pow(10000.0f, 2.0f * (i / 2) / embed_dim);
            // using the same exact formula from the paper

            if (i % 2 == 0)
                pe(pos, i) = std::sin(angle);
            else
                pe(pos, i) = std::cos(angle);
        }
    }
    
    return pe;
}