#pragma once
#include <vector>
#include <memory>
#include "layers/linear.hpp"
#include "tensor.hpp"

class FeedForward {
public:
    // layer_sizes: e.g., {input_dim, hidden1, hidden2, output_dim}
    FeedForward(const std::vector<int>& layer_sizes);

    // Forward pass: x (batch_size x input_dim)
    Tensor forward(const Tensor& x) const;

private:
    std::vector<std::unique_ptr<Linear>> layers;

    // ReLU activation
    Tensor relu(const Tensor& x) const;
};
