#pragma once
#include "tensor.hpp"
#include "layers/linear.hpp"
#include <memory>

class FeedForward {
public:
    FeedForward(int embed_dim, int hidden_dim);
    
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    
    void zero_grad();
    std::vector<std::shared_ptr<Tensor>> parameters() const;
    

    void get_parameters() const;
    void get_gradients() const;
    void get_layer_info() const;

private:
    Linear fc1;
    Linear fc2;
};

