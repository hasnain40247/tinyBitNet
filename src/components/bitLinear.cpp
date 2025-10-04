#include "components/bitLinear.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>

BitLinear::BitLinear(int in_features_, int out_features_, bool use_bias_)
    : in_features(in_features_), out_features(out_features_), use_bias(use_bias_), 
      layernorm(in_features_)
{
    initialize_parameters();
}

void BitLinear::initialize_parameters() {
    double limit = std::sqrt(6.0 / (in_features + out_features));
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-limit, limit);

    Eigen::MatrixXd W_data(in_features, out_features);
    for (int i = 0; i < W_data.size(); ++i) {
        W_data.data()[i] = dist(gen);
    }
    W = std::make_shared<Tensor>(W_data, true); 

    if (use_bias) {
        Eigen::MatrixXd b_data(1, out_features);
        for (int i = 0; i < out_features; ++i) {
            b_data(0, i) = dist(gen);  
        }
        b = std::make_shared<Tensor>(b_data, true); 
    } else {
        b = nullptr;
    }
}


void BitLinear::zero_grad() {
     W->zero_grad();
    if (b)    
    b->zero_grad();
}

std::vector<std::shared_ptr<Tensor>> BitLinear::parameters() const {
    if (b) return {W, b};
    return {W};
}


std::shared_ptr<Tensor> BitLinear::forward(std::shared_ptr<Tensor> x) {
 
double Qb = 127.0;
double eps = 1e-8;
auto x_norm = layernorm.forward(x);
auto quant= x_norm->quantize(Qb,eps);
auto W_binarize= W->binarize();

double beta = W->data.cwiseAbs().mean();
double gamma = x_norm->data.cwiseAbs().maxCoeff();

auto y = quant->mm(W_binarize)->scale(beta * gamma / 127.0);
if (b) y = y->addB(b);


return y;

}
