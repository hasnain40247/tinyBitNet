#pragma once
#include <memory>
#include "data/tensor.hpp"

class BaseLinear {
public:
    virtual ~BaseLinear() = default;
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> parameters() const = 0;
    virtual void zero_grad() = 0;
};
