#pragma once
#include <Eigen/Dense>
#include <vector>

/**
 * Multi-Head Attention.
 * 
 * Dimensions:
 *  - Input X: [seq_len, embed_dim]
 * TODO: Extend into batch processing by adding another dimension.
 *  - W_Q, W_K, W_V: [embed_dim, embed_dim]
 *  - For each head h we have:
 *      Q_h, K_h, V_h: [seq_len, head_dim]   where head_dim = embed_dim / num_heads (Used middlecols which apparently slices the matrix into blocks)
 *      scores: [seq_len, seq_len] - this is where we compute the similarity between the queries and the keys, normalize it with the dimension of the head
 *      attn:   [seq_len, seq_len] - the actual attention is just scores as weighted by the values.
 *  - Concatenated heads: [seq_len, embed_dim] - we then concatenate all of it into one matrix.
 *  - W_O: [embed_dim, embed_dim]
 *  - Output: [seq_len, embed_dim]
 */
class MultiHeadAttention {
public:
    int embed_dim;   // embedding dimension (each input has its own embedding)
    int num_heads;   // number of attention heads (these divide the Weight matrices)
    int head_dim;    // dimension per head (d_k = embed_dim / num_heads)

    Eigen::MatrixXd W_Q; // query matrix [embed_dim, embed_dim]
    Eigen::MatrixXd W_K; // key matrix   [embed_dim, embed_dim]
    Eigen::MatrixXd W_V; // value matrix [embed_dim, embed_dim]
    Eigen::MatrixXd W_O; // output matrix [embed_dim, embed_dim]

    /**
     * Constructor.
     * Initializes weight matrices with random values. I am sticking with Xavier's random initialization.
     * Upon reading through very few resources, it seems like a good choice without risking vanishing gradients. 
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
     * Forward pass of multi-head attention. At the time of writing this, I already know backpropagation is going to be a pain to implement.
     * @param X : input sequence embeddings [seq_len, embed_dim]
     * @param mask : optional attention mask [seq_len, seq_len] (default nullptr but I'll set it to inf like the book says)
     * @return output embeddings [seq_len, embed_dim]
     */
    Eigen::MatrixXd forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd* mask = nullptr);
};
