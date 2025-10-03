#pragma once
#include "../data/tensor.hpp"
#include <memory>

class Linear {
public:
    Linear(int in_features, int out_features);
    
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    
    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> b;
    
    void zero_grad();
    
    std::vector<std::shared_ptr<Tensor>> parameters() const;

private:
    int in_features;
    int out_features;
    
    void initialize_parameters();
};