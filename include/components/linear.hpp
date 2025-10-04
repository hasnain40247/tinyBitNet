#pragma once
#include "../data/tensor.hpp"
#include <memory>
#include "abstract/baseLinear.hpp"


class Linear: public BaseLinear {
public:
    Linear(int in_features, int out_features, bool use_bias = true);
    
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;
    
    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> b;
    
    void zero_grad() override;
    
    std::vector<std::shared_ptr<Tensor>> parameters() const override;

private:
    int in_features;
    int out_features;
    bool use_bias;
    
    void initialize_parameters();
};