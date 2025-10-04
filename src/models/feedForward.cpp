#include "models/feedForward.hpp"
#include <iostream>
#include "components/linear.hpp"
#include "components/bitLinear.hpp"


FeedForward::FeedForward(int embed_dim, int hidden_dim, bool use_bitlinear_)
    : use_bitlinear(use_bitlinear_)
{
    if (use_bitlinear) {
        fc1 = std::make_shared<BitLinear>(embed_dim, hidden_dim);
        fc2 = std::make_shared<BitLinear>(hidden_dim, embed_dim);
    } else {
        fc1 = std::make_shared<Linear>(embed_dim, hidden_dim);
        fc2 = std::make_shared<Linear>(hidden_dim, embed_dim);
    }
}

std::shared_ptr<Tensor> FeedForward::forward(std::shared_ptr<Tensor> x) {

    
    auto hidden = fc1->forward(x);
    
    auto activated = Tensor::gelu(hidden);

    
    auto output = fc2->forward(activated);
    
    
    return output;
}

void FeedForward::zero_grad() {
    fc1->zero_grad();
    fc2->zero_grad();
}

std::vector<std::shared_ptr<Tensor>> FeedForward::parameters() const {
    auto params1 = fc1->parameters();
    auto params2 = fc2->parameters();
    params1.insert(params1.end(), params2.begin(), params2.end());
    return params1;
}


