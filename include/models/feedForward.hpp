#pragma once
#include "../data/tensor.hpp"
#include "components/linear.hpp"
#include "components/bitLinear.hpp"  
#include <memory>
#include <vector>
#include "abstract/baseLinear.hpp"


class FeedForward {
public:
    FeedForward(int embed_dim, int hidden_dim, bool use_bitlinear_ = false);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);

    void zero_grad();
    std::vector<std::shared_ptr<Tensor>> parameters() const;

private:
    bool use_bitlinear;
    std::shared_ptr<BaseLinear> fc1;  
    std::shared_ptr<BaseLinear> fc2;
};
