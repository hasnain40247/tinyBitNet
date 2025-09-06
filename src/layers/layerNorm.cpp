#include "layers/layerNorm.hpp"
#include <cmath>

LayerNorm::LayerNorm(int embed_dim, float eps)
    : embed_dim(embed_dim), epsilon(eps)
{
    gamma = TensorFloat::Ones(embed_dim);
    beta = TensorFloat::Zero(embed_dim);
}

Tensor LayerNorm::forward(const Tensor& x) {
    Tensor out = x;
    for (int i = 0; i < x.rows(); ++i) {
        TensorFloat row = x.row(i);
        float mean = row.mean();
        float var = (row.array() - mean).square().mean();
        out.row(i) = ((row.array() - mean) / std::sqrt(var + epsilon)).matrix().cwiseProduct(gamma) + beta;
    }
    return out;
}
