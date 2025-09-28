

#include "layers/feedForward.hpp"
#include "layers/optimizer.hpp"
#include <iostream>
#include "tensor.hpp"
#include "layers/attention.hpp"
#include "layers/linear.hpp"


int main() {

    // Create an Embedding layer; the constructor will call initialize_parameters()
    // Embedding embed(10, 8, 5, false);
    // std::vector<int> input_indices = {2, 5, 3, 2, 7};

    // // Forward pass
    // std::shared_ptr<Tensor> output = embed.forward(input_indices);


  

    // return 0;

    int seq_len   = 4;
    int embed_dim = 8;
    int num_heads = 2;

    MultiHeadAttention mha(embed_dim, num_heads);

    // --- Build a dummy input Tensor ---
    Eigen::MatrixXd x_data(seq_len, embed_dim);
    x_data.setRandom();  // values in [-1,1)
    auto x = std::make_shared<Tensor>(x_data, /*requires_grad=*/true);

    std::cout << "Input Tensor X (" << seq_len << " x " << embed_dim << "):\n";
    x->get_data();
    std::cout << "\n";

    
    
    return 0;
}
