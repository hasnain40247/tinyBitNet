#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "data/tensor.hpp"
#include "components/linear.hpp"   // <- use your Linear class
#include "abstract/baseLinear.hpp"

/**
 * Multi-Head Attention.
 * 
 * Dimensions:
 *  - Input X: [seq_len, embed_dim]
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
    int embed_dim;
    int num_heads;
    int head_dim;
    bool use_bitlinear;

    // Instead of raw Tensors, wrap them in Linear
    
    std::shared_ptr<BaseLinear> W_Q;
    std::shared_ptr<BaseLinear> W_K;
    std::shared_ptr<BaseLinear> W_V;
    std::shared_ptr<BaseLinear> W_O;

    MultiHeadAttention(int embed_dim_, int num_heads_, bool use_bitlinear_ = false);

    Eigen::MatrixXd softmax(const Eigen::MatrixXd& x);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x,
                                    std::shared_ptr<Tensor> mask = nullptr);

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        auto params = W_Q->parameters();
        auto k_params = W_K->parameters();
        auto v_params = W_V->parameters();
        auto o_params = W_O->parameters();

        params.insert(params.end(), k_params.begin(), k_params.end());
        params.insert(params.end(), v_params.begin(), v_params.end());
        params.insert(params.end(), o_params.begin(), o_params.end());
        return params;
    }

    void zero_grad() {
        W_Q->zero_grad();
        W_K->zero_grad();
        W_V->zero_grad();
        W_O->zero_grad();
    }
};
