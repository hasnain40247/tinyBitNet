#pragma once
#include "../data/tensor.hpp"
#include "../components/linear.hpp"
#include <memory>

class FeedForward {
public:
    FeedForward(int embed_dim, int hidden_dim);
    
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    
    void zero_grad();
    std::vector<std::shared_ptr<Tensor>> parameters() const;
    

private:
    Linear fc1;
    Linear fc2;
};

