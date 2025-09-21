#include "layers/feedForward.hpp"
#include <iostream>

FeedForward::FeedForward(int embed_dim, int hidden_dim)
    : fc1(embed_dim, hidden_dim), fc2(hidden_dim, embed_dim) {}

std::shared_ptr<Tensor> FeedForward::forward(std::shared_ptr<Tensor> x) {
    std::cout << "\n=== FeedForward Forward Pass ===" << std::endl;
    
    std::cout << "\n--- Input ---" << std::endl;
    x->get();
    
    auto hidden = fc1.forward(x);
    std::cout << "\n--- After FC1 ---" << std::endl;
    hidden->get();
    
    auto activated = Tensor::relu(hidden);
    std::cout << "\n--- After ReLU ---" << std::endl;
    activated->get();
    
    auto output = fc2.forward(activated);
    std::cout << "\n--- Final Output ---" << std::endl;
    output->get();
    
    std::cout << "=================================" << std::endl;
    
    return output;
}

void FeedForward::zero_grad() {
    fc1.zero_grad();
    fc2.zero_grad();
}

std::vector<std::shared_ptr<Tensor>> FeedForward::parameters() const {
    auto params1 = fc1.parameters();
    auto params2 = fc2.parameters();
    params1.insert(params1.end(), params2.begin(), params2.end());
    return params1;
}

void FeedForward::get_parameters() const {
    std::cout << "=== FeedForward Parameters ===" << std::endl;
    
    std::cout << "\n--- FC1 Weight ---" << std::endl;
    fc1.W->get_data();
    
    std::cout << "\n--- FC1 Bias ---" << std::endl;
    fc1.b->get_data();
    
    std::cout << "\n--- FC2 Weight ---" << std::endl;
    fc2.W->get_data();
    
    std::cout << "\n--- FC2 Bias ---" << std::endl;
    fc2.b->get_data();
    
    std::cout << "===============================" << std::endl;
}

void FeedForward::get_gradients() const {
    std::cout << "=== FeedForward Gradients ===" << std::endl;
    
    std::cout << "\n--- FC1 Weight Grad ---" << std::endl;
    fc1.W->get_grad();
    
    std::cout << "\n--- FC1 Bias Grad ---" << std::endl;
    fc1.b->get_grad();
    
    std::cout << "\n--- FC2 Weight Grad ---" << std::endl;
    fc2.W->get_grad();
    
    std::cout << "\n--- FC2 Bias Grad ---" << std::endl;
    fc2.b->get_grad();
    
    std::cout << "==============================" << std::endl;
}

void FeedForward::get_layer_info() const {
    std::cout << "=== FeedForward Layer Info ===" << std::endl;
    std::cout << "FC1: " << fc1.W->data.rows() << " -> " << fc1.W->data.cols() << std::endl;
    std::cout << "FC2: " << fc2.W->data.rows() << " -> " << fc2.W->data.cols() << std::endl;
    std::cout << "Total parameters: " << parameters().size() << std::endl;
    std::cout << "===============================" << std::endl;
}