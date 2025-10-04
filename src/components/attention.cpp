#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>
#include "components/attention.hpp"
#include "components/linear.hpp"
#include "components/bitLinear.hpp"

#include "data/tensor.hpp"

MultiHeadAttention::MultiHeadAttention(int embed_dim_, int num_heads_, bool use_bitlinear_)
    : embed_dim(embed_dim_),
      num_heads(num_heads_),
      head_dim(embed_dim_ / num_heads_),use_bitlinear(use_bitlinear_)
{
    if (embed_dim % num_heads != 0)
        throw std::invalid_argument("embed_dim must be divisible by num_heads");

    if (use_bitlinear) {
        W_Q = std::make_shared<BitLinear>(embed_dim_, embed_dim_);
        W_K = std::make_shared<BitLinear>(embed_dim_, embed_dim_);
        W_V = std::make_shared<BitLinear>(embed_dim_, embed_dim_);
        W_O = std::make_shared<BitLinear>(embed_dim_, embed_dim_);
    } else {
        W_Q = std::make_shared<Linear>(embed_dim_, embed_dim_);
        W_K = std::make_shared<Linear>(embed_dim_, embed_dim_);
        W_V = std::make_shared<Linear>(embed_dim_, embed_dim_);
        W_O = std::make_shared<Linear>(embed_dim_, embed_dim_);
    }
}


/**
 * Row-wise softmax over the last dimension.
 */
Eigen::MatrixXd MultiHeadAttention::softmax(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd y = x;
    for (int i = 0; i < x.rows(); i++) {
        double max_val = x.row(i).maxCoeff();
        Eigen::VectorXd exps = (x.row(i).array() - max_val).exp();
        y.row(i) = exps / exps.sum();
    }
    return y;
}

/**
 * Forward pass of multi-head attention.
 */
std::shared_ptr<Tensor> MultiHeadAttention::forward(std::shared_ptr<Tensor> x,
                                                    std::shared_ptr<Tensor> mask) {
    int seq_len = x->data.rows();

    // Projections through Linear layers
    auto Q = W_Q->forward(x); // [seq_len, embed_dim]
    auto K = W_K->forward(x); // [seq_len, embed_dim]
    auto V = W_V->forward(x); // [seq_len, embed_dim]

    std::vector<std::shared_ptr<Tensor>> head_outputs;

    for (int h = 0; h < num_heads; h++) {
        auto Q_h = Q->slice(h * head_dim, head_dim); // [seq_len, head_dim]
        auto K_h = K->slice(h * head_dim, head_dim); // [seq_len, head_dim]
        auto V_h = V->slice(h * head_dim, head_dim); // [seq_len, head_dim]

        // Attention scores: [seq_len, seq_len]
        auto scores = Q_h->mm(K_h->transpose())
                          ->scale(1.0 / std::sqrt((double)head_dim));

        if (mask != nullptr) {
            scores = scores->operator+(mask);  // add mask if provided
        }

        // Softmax along rows
        auto attn = scores->softmax();

        // Weighted sum: [seq_len, head_dim]
        auto head = attn->mm(V_h);
        head_outputs.push_back(head);
    }

    auto concat = Tensor::concat_cols(head_outputs);

    // Final output projection
    auto output = W_O->forward(concat); // [seq_len, embed_dim]
    return output;
}
