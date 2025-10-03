#pragma once
#include "../data/tensor.hpp"
#include <vector>
#include <memory>

class SGD {
public:
    SGD(const std::vector<std::shared_ptr<Tensor>>& parameters, double lr);
    
    void step();
    void zero_grad();

private:
    std::vector<std::shared_ptr<Tensor>> parameters;
    double learning_rate;
};

SGD::SGD(const std::vector<std::shared_ptr<Tensor>>& parameters, double lr)
    : parameters(parameters), learning_rate(lr) {}

void SGD::step() {
    for (auto& param : parameters) {
        if (param->requires_grad) {
            param->data -= learning_rate * param->grad;
        }
    }
}

void SGD::zero_grad() {
    for (auto& param : parameters) {
        param->zero_grad();
    }
}