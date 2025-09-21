#pragma once
#include "tensor.hpp"
#include <memory>

class Linear {
public:
    Linear(int in_features, int out_features);
    
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    
    // Parameters as tensors
    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> b;
    
    // Zero gradients for parameters
    void zero_grad();
    
    // Get parameters for optimizer
    std::vector<std::shared_ptr<Tensor>> parameters() const;

private:
    int in_features;
    int out_features;
    
    void initialize_parameters();
};