#pragma once
#include "../data/tensor.hpp"
#include <memory>
#include <vector>
#include "components/layerNorm.hpp"  
#include "abstract/baseLinear.hpp"

class BitLinear: public BaseLinear  {
public:
    BitLinear(int in_features, int out_features, bool use_bias = true);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;

    // Full-precision weights and optional bias
    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> b;

    LayerNorm layernorm;

    void zero_grad() override;

    std::vector<std::shared_ptr<Tensor>> parameters() const override;

private:
    int in_features;
    int out_features;
    bool use_bias;

    void initialize_parameters();

};
