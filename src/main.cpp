#include "layers/embedding.hpp"
#include "block.hpp"
#include <iostream>
#include <Eigen/Dense>

int main() {
    int vocab_size = 100;
    int embed_dim = 8;
    int max_seq = 10;
    int seq_len = 5;
    int num_heads = 2;

    // 1. Create embedding
    Embedding embed(vocab_size, embed_dim, max_seq, true);

    // 2. Sample input indices
    Eigen::VectorXi input_indices(seq_len);
    input_indices << 1, 5, 3, 7, 2;

    // 3. Forward pass through embedding layer
    Eigen::MatrixXd X = embed.forward(input_indices); // shape [seq_len, embed_dim]

    // 4. Create Transformer block
    TransformerBlock block(embed_dim, num_heads);

    // 5. Forward pass through transformer
    Eigen::MatrixXd output = block.forward(X);

    std::cout << "Output shape: [" << output.rows() << ", " << output.cols() << "]\n";
    std::cout << "Output:\n" << output << std::endl;

    return 0;  
}
