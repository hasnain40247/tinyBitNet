

// #include "block.hpp"

// /**
//  * @brief Constructor for TransformerBlock
//  * 
//  * Initializes LayerNorm and Multi-Head Attention with given embedding dimension and number of heads.
//  */
// TransformerBlock::TransformerBlock(int embed_dim_, int num_heads_)
//     : embed_dim(embed_dim_), num_heads(num_heads_),
//       ln1(embed_dim_),                     // LayerNorm
//       mha(embed_dim_, num_heads_)          // Multi-head attention
// {

// }

// /**
//  * @brief Forward pass for TransformerBlock
//  * 
//  * Steps:
//  * 1. LayerNorm
//  * 2. Multi-Head Attention
//  * 
//  * @param X Input tensor [seq_len, embed_dim]
//  * @param mask Optional attention mask [seq_len, seq_len]
//  * @return Output tensor [seq_len, embed_dim]
//  */
// Eigen::MatrixXd TransformerBlock::forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd* mask) {
//     // 1. LayerNorm
//     Eigen::MatrixXd X_norm = ln1.forward(X);

//     // 2. Multi-Head Attention
//     Eigen::MatrixXd attn_out = mha.forward(X_norm, mask);

//     return X + attn_out;
// }
