#include "components/linear.hpp"
#include <random>
#include <cmath>

Linear::Linear(int in_features_, int out_features_, bool use_bias_)
    : in_features(in_features_), out_features(out_features_), use_bias(use_bias_) {
    initialize_parameters();
}

void Linear::initialize_parameters() {
    double limit = std::sqrt(6.0 / (in_features + out_features));
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-limit, limit);
    
    // Initialize W
    Eigen::MatrixXd W_data(in_features, out_features);
    for (int i = 0; i < W_data.size(); ++i)
        W_data.data()[i] = dist(gen);
    W = std::make_shared<Tensor>(W_data, true); 
    
    // Initialize b only if use_bias = true
    if (use_bias) {
        Eigen::MatrixXd b_data(1, out_features);
        for (int i = 0; i < out_features; ++i)
            b_data(0, i) = dist(gen);
        b = std::make_shared<Tensor>(b_data, true);
    } else {
        b = nullptr;
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> x) {
    auto y = x->mm(W);
    if (use_bias && b)
        y = y->addB(b);  
    return y;
}


void Linear::zero_grad() {
    W->zero_grad();
    if (use_bias && b)
        b->zero_grad();
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() const {
    if (use_bias && b)
        return {W, b};
    return {W};
}
