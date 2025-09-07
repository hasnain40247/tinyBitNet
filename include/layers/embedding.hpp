#pragma once

#include "tensor.hpp"
#include <Eigen/Dense>
#include <iostream>

/**
 * @brief Embedding Module for token and positional embeddings.
 *
 * This one has both learnable token embeddings and optional
 * sinusoidal or learnable positional embeddings. Given a sequence of
 * token indices, it returns their corresponding embeddings with added
 * positional stuff.
 *
 * Dimensions:
 *  - token_embeddings: [vocab_size, embed_dim]
 *  - positional_embeddings: [max_seq_len, embed_dim]
 *  - Input indices: [seq_len]
 *  - Output: [seq_len, embed_dim]
 */
class Embedding {
public:
    /**
     * @brief Constructor for the Embedding layer.
     * 
     * @param vocab_size  Number of unique tokens in the vocabulary.
     * @param embed_dim   Dimensionality of the embeddings.
     * @param max_seq     Maximum sequence length (used for positional embeddings).
     * @param sinusoidal  If true, use sinusoidal positional embeddings; 
     *                    otherwise, learnable positional embeddings.
     */
    Embedding(int vocab_size, int embed_dim, int max_seq, bool sinusoidal);

    /**
     * @brief Forward pass for the embedder.
     * 
     * Converts a sequence of token indices into embeddings and adds positional embeddings.
     * 
     * @param input_indices Eigen::VectorXi of token indices of shape [seq_len].
     * @return Eigen::MatrixXd of embeddings of shape [seq_len, embed_dim].
     */
    Eigen::MatrixXd forward(const Eigen::VectorXi& input_indices);



private:
    Eigen::MatrixXd token_embeddings;      // Learnable token embeddings: [vocab_size, embed_dim]
    Eigen::MatrixXd positional_embeddings; // Positional embeddings: [max_seq_len, embed_dim]
    int vocab_size;               // vocabulary size
    int embed_dim;                // Embedding dimension
    int max_seq;                  // Maximum sequence length

    /**
     * @brief Generates positional embeddings.
     * 
     * Can be sinusoidal or learnable depending on the constructor.
     * 
     * @param max_seq   Maximum sequence length
     * @param embed_dim Embedding dimension
     * @return Eigen::MatrixXd of shape [max_seq, embed_dim]
     */
    Eigen::MatrixXd positional_embedding(int max_seq, int embed_dim);
};
