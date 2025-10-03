#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "../data/tensor.hpp"


/**
 * Layer Normalization.
 * 
 * Normalizes input across the last dimension (hidden size).
 * Formula: 
 *   y = (x - mean) / sqrt(var + eps) * gamma + beta
 * 
 * Dimensions:
 *  - Input:  [seq_len, embed_dim]
 *  - Gamma:  [1, embed_dim]
 *  - Beta:   [1, embed_dim]
 *  - Output: [seq_len, embed_dim]
 */
class LayerNorm {
public:
    int embed_dim;   // size of hidden dimension being normalized
    double eps;      // numerical stability constant

    // Learnable parameters
    std::shared_ptr<Tensor> gamma; // scale [1, embed_dim]
    std::shared_ptr<Tensor> beta;  // shift [1, embed_dim]

    /**
     * Constructor.
     * Initializes gamma to ones and beta to zeros.
     * @param embed_dim_ : size of hidden dimension
     * @param eps_ : epsilon for numerical stability (default 1e-5)
     */
    LayerNorm(int embed_dim_, double eps_ = 1e-5);

    /**
     * Forward pass.
     * Applies layer normalization over last dimension.
     * @param x : input tensor [seq_len, embed_dim]
     * @return normalized output [seq_len, embed_dim]
     */
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);

    /**
     * Get trainable parameters
     */
    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {gamma, beta};
    }

    /**
     * Zero gradients
     */
    void zero_grad() {
        gamma->zero_grad();
        beta->zero_grad();
    }
};
