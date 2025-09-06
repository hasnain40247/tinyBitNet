#include "layers/linear.hpp"
#include <random>
#include <cmath>

Linear::Linear(int in_features, int out_features) {
    // Xavier uniform initialization
    float limit = std::sqrt(6.0f / (in_features + out_features));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-limit, limit);

    // Initialize weights
    W = Tensor(out_features, in_features);
    for (int i = 0; i < W.rows(); ++i)
        for (int j = 0; j < W.cols(); ++j)
            W(i, j) = dist(gen);

    // Initialize bias
    b = TensorFloat::Zero(out_features);
}

Tensor Linear::forward(const Tensor& x) const {
    // x: (batch_size x in_features)
    // W: (out_features x in_features)
    // b: (out_features)
    // output: (batch_size x out_features)
    
    // Multiply x by W^T to get correct output shape
    Tensor y = x * W.transpose();  // (batch_size x out_features)
    
    // Broadcast bias over rows
    y.rowwise() += b.transpose();
    return y;
}


