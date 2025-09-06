#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include "layers/embedding.hpp"
#include "tensor.hpp"

TEST_CASE("Embedding forward produces non-empty output", "[embedding]") {
    int vocab_size = 50;
    int embed_dim = 8;
    int max_seq = 10;
    bool sinusoidal = true;

    Embedding emb(vocab_size, embed_dim, max_seq, sinusoidal);

    TensorInt input_indices(5);          // sequence length 5
    input_indices << 3, 12, 7, 25, 0;   // some token IDs

    std::cout << "Input indices:\n" << input_indices.transpose() << "\n\n";

    Tensor output = emb.forward(input_indices);

    std::cout << "Embedding output:\n" << output << "\n\n";

    REQUIRE(output.size() > 0);          // output has content
}

TEST_CASE("Embedding output shape is correct", "[embedding]") {
    int vocab_size = 50;
    int embed_dim = 8;
    int max_seq = 10;
    bool sinusoidal = false;

    Embedding emb(vocab_size, embed_dim, max_seq, sinusoidal);

    TensorInt input_indices(5);
    input_indices << 3, 12, 7, 25, 0;

    Tensor output = emb.forward(input_indices);

    std::cout << "Embedding forward output shape: "
              << output.rows() << " x " << output.cols() << "\n";

    REQUIRE(output.rows() == input_indices.size());  // seq_len
    REQUIRE(output.cols() == embed_dim);            // embedding dimension
}
