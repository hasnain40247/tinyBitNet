#include "layers/linear.hpp"
#include <random>
#include <cmath>

Linear::Linear(int in_features_, int out_features_)
    : in_features(in_features_), out_features(out_features_) {
    initialize_parameters();
}

void Linear::initialize_parameters() {
    // Xavier initialization
    double limit = std::sqrt(6.0 / (in_features + out_features));
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-limit, limit);
    
    // Initialize weight matrix
    Eigen::MatrixXd W_data(in_features, out_features);
    for (int i = 0; i < W_data.size(); ++i) {
        W_data.data()[i] = dist(gen);
    }
    W = std::make_shared<Tensor>(W_data, true); // requires_grad = true
    

    Eigen::MatrixXd b_data(1, out_features);
    for (int i = 0; i < out_features; ++i) {
        b_data(0, i) = dist(gen);
    }
    b = std::make_shared<Tensor>(b_data, true); // requires_grad = true
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> x) {
    // x: [seq_len, in_features]
    // W: [in_features, out_features]
    // b: [1, out_features] (will broadcast)
    
    auto xW = x->mm(W);  // [seq_len, out_features]
    auto result=xW->addB(b);    
    return result;
}

void Linear::zero_grad() {
    W->zero_grad();
    b->zero_grad();
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() const{
    return {W, b};
}