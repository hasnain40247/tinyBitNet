#pragma once
#include <Eigen/Dense>
#include "tensor.hpp"

/**
 * @brief Layer Normalization.
 * 
 * Okay quick refresher. For theinput tensor X of shape [seq_len, embed_dim], it normalizes each row:
 *                  {centering the value} / scaling it
 *      x_norm[i] = (x[i] - mean_i) / sqrt(var_i + epsilon) * gamma + beta
 * 
 * Dimensions:
 *  - Input X: [seq_len, embed_dim]
 *  - gamma, beta: [embed_dim] - these are learnt.
 *  - Output: [seq_len, embed_dim] (same as input)
 */
class LayerNorm {
public:
    int embed_dim;               // Embedding dimension
    Eigen::VectorXd gamma;       // Scale parameter: [embed_dim]
    Eigen::VectorXd beta;        // Shift parameter: [embed_dim]
    double epsilon;              // Numerical stability

    /**
     * @brief Constructor.
     * 
     * Initializes gamma to ones and beta to zeros by default.
     * 
     * @param embed_dim Feature dimension 
     * @param eps Small epsilon to avoid division by zero 
     */
   LayerNorm(int embed_dim, double eps = 1e-5);

    /**
     * @brief Forward pass of layer normalization.
     * 
     * Normalizes each row of the input tensor along the embedding dimension.
     * 
     * @param x Input tensor of shape [seq_len, embed_dim]
     * @return Normalized tensor of same shape [seq_len, embed_dim]
     */
  Eigen::MatrixXd forward(const Eigen::MatrixXd& x);
};
