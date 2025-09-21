// #include "layers/embedding.hpp"
// #include "block.hpp"
// #include <iostream>
// #include <Eigen/Dense>
// #include "layers/feedForward.hpp"

// int main() {
//     // int vocab_size = 100;
//     // int embed_dim = 8;
//     // int max_seq = 10;
//     // int seq_len = 5;
//     // int num_heads = 2;

//     // // 1. Create embedding
//     // Embedding embed(vocab_size, embed_dim, max_seq, true);

//     // // 2. Sample input indices
//     // Eigen::VectorXi input_indices(seq_len);
//     // input_indices << 1, 5, 3, 7, 2;

//     // // 3. Forward pass through embedding layer
//     // Eigen::MatrixXd X = embed.forward(input_indices); // shape [seq_len, embed_dim]

//     // // 4. Create Transformer block
//     // TransformerBlock block(embed_dim, num_heads);

//     // // 5. Forward pass through transformer
//     // Eigen::MatrixXd output = block.forward(X);

//     // std::cout << "Output shape: [" << output.rows() << ", " << output.cols() << "]\n";
//     // std::cout << "Output:\n" << output << std::endl;

//     int seq_len    = 4;   // number of “tokens” (rows)
//     int embed_dim  = 6;   // input/output dimension
//     int hidden_dim = 12;  // hidden layer size inside FeedForward

//     // 1. Create a random input matrix [seq_len, embed_dim]
//     Eigen::MatrixXd X(seq_len, embed_dim);
//     X.setRandom();   // values in [-1, 1]
//     std::cout << "Input X:\n" << X << "\n\n";

//     // 2. Instantiate FeedForward
//     FeedForward ffn(embed_dim, hidden_dim);

//     // 3. Forward pass
//     Eigen::MatrixXd Y = ffn.forward(X);

//     // 4. Print output
//     std::cout << "Output Y shape: [" << Y.rows()
//               << ", " << Y.cols() << "]\n";
//     std::cout << "Output Y:\n" << Y << std::endl;


//     return 0;  
// }


#include "layers/feedForward.hpp"
#include "layers/optimizer.hpp"
#include <iostream>
#include "tensor.hpp"
int main() {
    // // Create model
    FeedForward model(4, 8);
    
    // // Create optimizer
    auto params = model.parameters();
    SGD optimizer(params, 0.001); // learning rate = 0.001
    
    // Create dummy input data [seq_len=10, embed_dim=512]
    Eigen::MatrixXd input_data = Eigen::MatrixXd::Random(3, 4);

    Eigen::MatrixXd target_data = Eigen::MatrixXd::Random(3, 4);
    auto target = std::make_shared<Tensor>(target_data, false);
    auto input = std::make_shared<Tensor>(input_data, false); // input doesn't need gradients
    input->get();
   
    model.get_layer_info();      // Show architecture
    model.get_parameters();      // Show all weights and biases  
    model.get_gradients(); 

    auto output = model.forward(input);

    auto diff = Tensor::add(output, 
                               std::make_shared<Tensor>(-target->data, false));
        
    double loss = diff->data.array().square().mean();


    std::cout << "\n--- Difference (output - target) ---" << std::endl;
    diff->get();

  
    std::cout << "\n--- Loss Value ---" << std::endl;
    std::cout << "MSE Loss: " << loss << std::endl;
    std::cout << "=========================" << std::endl;


        //     // Manual backward pass for loss (in real implementation, you'd have a Loss class)
    diff->grad = 2.0 * diff->data / (diff->data.rows() * diff->data.cols());
    diff->backward();
    std::cout << "\n=== After Backward Pass ===" << std::endl;
    model.get_gradients();

    optimizer.step();

    // // Create dummy target
    // Eigen::MatrixXd target_data = Eigen::MatrixXd::Random(10, 512);
    // auto target = std::make_shared<Tensor>(target_data, false);

    
    // // Training loop
    // for (int epoch = 0; epoch < 2; ++epoch) {
    //     // Zero gradients
    //     optimizer.zero_grad();
        
    //     // Forward pass
    //     auto output = model.forward(input);
        
    //     // Simple MSE loss: loss = ||output - target||^2
    //     auto diff = Tensor::add(output, 
    //                            std::make_shared<Tensor>(-target->data, false));
        
    //     // For simplicity, let's just compute mean of squared elements
    //     double loss = diff->data.array().square().mean();
        
    //     // Manual backward pass for loss (in real implementation, you'd have a Loss class)
    //     diff->grad = 2.0 * diff->data / (diff->data.rows() * diff->data.cols());
    //     diff->backward();
        
    //     // Update parameters
    //     optimizer.step();
        
    //     if (epoch % 10 == 0) {
    //         std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
    //     }
    // }
    
    return 0;
}

// Compilation example:
// g++ -std=c++17 -I/path/to/eigen -O2 example_usage.cpp tensor.cpp linear_autograd.cpp -o autograd_example