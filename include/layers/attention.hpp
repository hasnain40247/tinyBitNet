#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "tensor.hpp"  // assuming you have your Tensor class here

/**
 * Multi-Head Attention.
 * 
 * Dimensions:
 *  - Input X: [seq_len, embed_dim]
 * TODO: Extend into batch processing by adding another dimension.
 *  - W_Q, W_K, W_V: [embed_dim, embed_dim]
 *  - For each head h we have:
 *      Q_h, K_h, V_h: [seq_len, head_dim]   where head_dim = embed_dim / num_heads
 *      scores: [seq_len, seq_len]
 *      attn:   [seq_len, seq_len]
 *  - Concatenated heads: [seq_len, embed_dim]
 *  - W_O: [embed_dim, embed_dim]
 *  - Output: [seq_len, embed_dim]
 */
class MultiHeadAttention {
public:
    int embed_dim;   // embedding dimension (each input has its own embedding)
    int num_heads;   // number of attention heads
    int head_dim;    // dimension per head (d_k = embed_dim / num_heads)

    // Trainable weight tensors
    std::shared_ptr<Tensor> W_Q; // query matrix [embed_dim, embed_dim]
    std::shared_ptr<Tensor> W_K; // key matrix   [embed_dim, embed_dim]
    std::shared_ptr<Tensor> W_V; // value matrix [embed_dim, embed_dim]
    std::shared_ptr<Tensor> W_O; // output matrix [embed_dim, embed_dim]

    /**
     * Constructor.
     * Initializes weight tensors with random values (Xavier).
     * @param embed_dim_ : embedding dimension
     * @param num_heads_ : number of attention heads
     */
    MultiHeadAttention(int embed_dim_, int num_heads_);

    /**
     * Row-wise softmax over the last dimension.
     * @param x : [rows, cols] matrix
     * @return row-normalized matrix (softmax across columns per row)
     */
    Eigen::MatrixXd softmax(const Eigen::MatrixXd& x);

    /**
     * Forward pass of multi-head attention.
     * @param X : input sequence embeddings [seq_len, embed_dim]
     * @param mask : optional attention mask [seq_len, seq_len] (default nullptr)
     * @return output embeddings [seq_len, embed_dim]
     */
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> mask = nullptr);

    /**
     * Get trainable parameters
     */
    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {W_Q, W_K, W_V, W_O};
    }

    /**
     * Zero gradients
     */
    void zero_grad() {
        W_Q->zero_grad();
        W_K->zero_grad();
        W_V->zero_grad();
        W_O->zero_grad();
    }

    private:
    void initialize_parameters();

};
