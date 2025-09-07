#include "layers/layerNorm.hpp"
#include <cmath>

/**
* @brief Constructor.
* 
* Initializes gamma to ones and beta to zeros by default.
* 
* @param embed_dim Feature dimension 
* @param eps Small epsilon to avoid division by zero 
*/
LayerNorm::LayerNorm(int embed_dim, double eps)
    : embed_dim(embed_dim), epsilon(eps)
{
    gamma = Eigen::VectorXd::Ones(embed_dim);  // use double
    beta  = Eigen::VectorXd::Zero(embed_dim);  // use double
}

/**
* @brief Forward pass of layer normalization.
* 
* Normalizes each row of the input tensor along the embedding dimension.
* 
* @param x Input matrix of shape [seq_len, embed_dim]
* @return Normalized matrix of same shape [seq_len, embed_dim]
*/
Eigen::MatrixXd LayerNorm::forward(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd out = x;
    for (int i = 0; i < x.rows(); ++i) {
        Eigen::VectorXd row = x.row(i);
        double mean = row.mean();
        double var  = (row.array() - mean).square().mean();
        out.row(i) = ((row.array() - mean) / std::sqrt(var + epsilon)).matrix().cwiseProduct(gamma) + beta;
    }
    return out;
}
