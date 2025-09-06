#pragma once
#include <Eigen/Dense>
#include "tensor.hpp"


class LayerNorm {
public:
    int embed_dim;               // feature dimension
    TensorFloat gamma;       // scale parameter
    TensorFloat beta;        // shift parameter
    float epsilon;               // small number for numerical stability

    // Constructor
    LayerNorm(int embed_dim, float eps = 1e-5);

    // Forward pass
    // Input: (seq_len, embed_dim)
    // Output: normalized tensor of same shape
    Tensor forward(const Tensor& x);
};
