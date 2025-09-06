#include "layers/feedForward.hpp"
#include <algorithm>

// Constructor: create Linear layers
FeedForward::FeedForward(const std::vector<int>& layer_sizes) {
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        layers.push_back(std::make_unique<Linear>(layer_sizes[i], layer_sizes[i+1]));
    }
}

// ReLU activation
Tensor FeedForward::relu(const Tensor& x) const {
    Tensor y = x;
    for (int i = 0; i < y.rows(); ++i)
        for (int j = 0; j < y.cols(); ++j)
            y(i,j) = std::max(0.0f, y(i,j));
    return y;
}

// Forward pass
Tensor FeedForward::forward(const Tensor& x) const {
    Tensor out = x;
    for (size_t i = 0; i < layers.size(); ++i) {
        out = layers[i]->forward(out);
        // Apply ReLU to all layers except the last
        if (i < layers.size() - 1) {
            out = relu(out);
        }
    }
    return out;
}
