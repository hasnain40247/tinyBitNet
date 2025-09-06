#pragma once
#include <Eigen/Dense>
#include "tensor.hpp"

class Linear {
public:
    Linear(int in_features, int out_features);

    // Forward pass: x (batch_size x in_features)
    Tensor forward(const Tensor& x) const;

    const Tensor& getWeight() const { return W; }
    const TensorFloat& getBias() const { return b; }

private:
    Tensor W;        // (out_features x in_features)
    TensorFloat b;   // (out_features)
};
