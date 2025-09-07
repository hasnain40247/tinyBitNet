#pragma once
#include "layers/attention.hpp"
#include "layers/layerNorm.hpp"
#include <Eigen/Dense>

/**
 * @brief Transformer Block (TODO: A bunch of stuff to add)
 * 
 * This class implements a single Transformer block with:
 *  - Pre-LayerNorm
 *  - Multi-Head Attention
 *  
 * 
 * Input / output shapes:
 *  - Input:  [seq_len, embed_dim]
 *  - Output: [seq_len, embed_dim]
 * 
 * Forward pass:
 * 1. LayerNorm on input
 * 2. Multi-Head Attention
 */
class TransformerBlock {
public:
    int embed_dim;  // Embedding dimension 
    int num_heads;  // attention heads

    LayerNorm ln1;           // LayerNorm
    MultiHeadAttention mha;  // Attention layer

    /**
     * @brief Constructor
     * 
     * Initializes LayerNorm and Multi-Head Attention.
     * 
     * @param embed_dim Embedding dimension
     * @param num_heads Number of attention heads
     */
    TransformerBlock(int embed_dim, int num_heads);

    /**
     * @brief Forward pass
     * 
     * Computes a single Transformer block forward pass with attention.
     * 
     * @param X Input tensor of shape [seq_len, embed_dim]
     * @param mask Optional attention mask [seq_len, seq_len]. Default is nullptr.
     * @return Output tensor of shape [seq_len, embed_dim]
     */
    Eigen::MatrixXd forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd* mask = nullptr);
};
